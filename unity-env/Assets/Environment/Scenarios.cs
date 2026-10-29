using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using MotionEnv.AvatarLayer;
using Sora.Embodiment.Avatar;
using UniVRM10;
using UnityEngine;

namespace MotionEnv.Environment
{
    /// <summary>One scenario run's measurements.</summary>
    public sealed class ScenarioResult
    {
        public string Avatar = "", Scenario = "", Step = "";
        public HelpDial Help;
        public bool Passed;
        public string Reason = "";
        public readonly SortedDictionary<string, double> Metrics = new();
        public HelpLog HelpUsed = new();
        public List<string> Events = new();
        /// <summary>A1: whether hair and clothing stayed out of the body (null if not checked).</summary>
        public bool? ClippingPassed;
        public string ClippingNote = "";

        public override string ToString() =>
            $"{Avatar} / {Scenario} / {Step}: {(Passed ? "pass" : "FAIL")} ({Reason}); " +
            string.Join(", ", Metrics.Select(m => $"{m.Key} {m.Value:0.###}"));
    }

    /// <summary>
    /// The fixed 60 s scenarios (roadmap MA): stand, walk, sit, push, fall, get up. Each runs the
    /// physics body at 120 Hz, driven by a motion source (the stand-in motion until M1), and
    /// measures the gates: falls, pose error, jitter, foot slide, travel, rest gap and
    /// penetration after a fall, settling time, cost, help used and hair or skirt clipping.
    /// </summary>
    public sealed class ScenarioRunner
    {
        public static readonly string[] Names = { "stand", "walk", "run", "sit", "push", "fall", "get up" };
        /// <summary>Walking and running speeds, canonical metres a second (SORA's 300 and 800
        /// px/s are about 0.9 and 2.4 m/s on Alice).</summary>
        public const float WalkSpeed = 0.5f, RunSpeed = 2.4f;
        public const float Dt = 1f / 120f;
        public float Seconds = 60f;
        /// <summary>Simulate hair and clothing and measure clipping (roadmap A1).</summary>
        public bool CheckClipping = true;
        /// <summary>Diagnostics: called after every physics step with the time, the body, the
        /// fall controller, the surfaces and the wanted pose.</summary>
        public Action<float, PhysicsBody, FallController, DesktopSurfaces, CanonicalPose>? Trace;

        public IEnumerator Run(string avatar, string scenario, string step, HelpDial help, Action<ScenarioResult> done)
        {
            var result = new ScenarioResult { Avatar = avatar, Scenario = scenario, Step = step, Help = help };
            var previousMode = Physics.simulationMode;
            Physics.simulationMode = SimulationMode.Script;
            Physics.gravity = new Vector3(0, -9.81f, 0);
            Vrm10Instance instance = null!;
            yield return TestAvatars.Load(avatar, i => instance = i);
            var surfaces = DesktopSurfaces.Create(new DesktopSurfaces.Surface("taskbar", new Vector3(0, -0.05f, 0), new Vector3(400, 0.1f, 400)));
            var animator = instance.GetComponent<Animator>();
            var standHeight = 0f;
            if (scenario == "fall")
            {
                // She stands on a window's top edge, half her height above the taskbar; the window closes.
                var probeFit = new AvatarFit(animator);
                standHeight = 0.5f * probeFit.Height;
                surfaces.Add(new DesktopSurfaces.Surface("window", new Vector3(0, standHeight - 0.05f, 0), new Vector3(0.8f * probeFit.Height, 0.1f, 0.8f)));
                instance.transform.position += Vector3.up * standHeight;
            }
            var fit = new AvatarFit(animator);
            var body = new PhysicsBody(fit, groundY: standHeight);
            var falls = new FallController(body, surfaces) { ExpectFlight = scenario == "run" && StandInMotion.HasFlight(RunSpeed) };
            body.Help = help;
            falls.NormalHelp = help;
            AvatarSecondaryMotionTuner? tuner = null;
            if (CheckClipping) { tuner = instance.gameObject.AddComponent<AvatarSecondaryMotionTuner>(); tuner.Configure(instance); }
            var clipping = CheckClipping ? new ClippingProbe(instance, body) : null;

            var motion = new StandInMotion();
            var wanted = new CanonicalPose();
            motion.Stand(0, wanted);
            if (scenario == "sit")
            {
                var seatTop = fit.WorldSpace(fit.HipsFor(new Vector3(0, 0.47f, 0))).y - 0.06f * fit.Height;
                var seatZ = fit.WorldSpace(fit.HipsFor(new Vector3(0, 0.47f, -0.33f))).z;
                surfaces.Add(new DesktopSurfaces.Surface("seat", new Vector3(0, seatTop - 0.05f, seatZ - 0.05f * fit.Height),
                    new Vector3(0.6f * fit.Height, 0.1f, 0.3f * fit.Height)));
            }
            var rest = fit.Fold(wanted);
            var hipsStart = fit.WorldSpace(fit.HipsFor(wanted.Root));
            if (scenario == "get up")
            {
                // Lying on her back on the taskbar, loose.
                motion.Relaxed(wanted);
                var lying = Quaternion.Euler(-90, 0, 0) * fit.Root.rotation;
                body.Teleport(new Vector3(hipsStart.x, 0.11f * fit.Height, hipsStart.z), lying, fit.Fold(wanted));
            }
            else body.Teleport(hipsStart, fit.Root.rotation, rest);
            Physics.SyncTransforms();
            if (scenario == "get up") falls.StartResting(0);
            var headRest = body.Link(Canonical.Index("head")).transform.position.y - standHeight;

            var clock = new Stopwatch();
            var poseErrors = new List<float>();
            var jitter = 0f;
            var wobble = new Wobble();
            var footSlide = 0f;
            var plantedAt = new Vector3?[2];
            var feet = new[] { body.Link(Canonical.Index("leftFoot")), body.Link(Canonical.Index("rightFoot")) };
            var footColliders = feet.Select(f => f.GetComponent<Collider>()).ToArray();
            var lowestHead = float.MaxValue;
            var pushes = 0;
            var stoodAt = -1f;
            var steps = Mathf.RoundToInt(Seconds / Dt);
            var maxSpeed = 0f;
            Vector3? hipsAtImpact = null;
            var targets = new Quaternion[fit.Map.Joints.Count];
            (float Lowest, float Deepest)? mesh = null, meshAll = null;
            var meshPart = "";
            float runPhase = 0f, runDistance = 0f;
            var fastestLink = "";
            var lastSecondMin = new Dictionary<string, Vector3>();
            var lastSecondMax = new Dictionary<string, Vector3>();
            for (var s = 0; s < steps; s++)
            {
                var t = s * Dt;
                switch (scenario)
                {
                    case "walk": motion.Walk(t, WalkSpeed, wanted); break;
                    case "run":
                    {
                        // Speeding up over the first second, as a person does.
                        var v = RunSpeed * Mathf.Clamp01(t / 1f);
                        runPhase += Dt / StandInMotion.GaitPeriod(v);
                        runDistance += v * Dt;
                        motion.Gait(runPhase, v, wanted);
                        wanted.Root = new Vector3(0, wanted.Root.y, runDistance);
                        break;
                    }
                    case "sit": motion.Sit(Mathf.Clamp01((t - 2f) / 2f), wanted); break;
                    default: motion.Stand(t, wanted); break;
                }
                var pose = falls.Step(Dt, t, wanted);
                fit.Fold(pose, targets);
                body.SetTargets(targets);
                if (falls.Phase == FallPhase.Supported)
                {
                    body.BalanceTargetPosition = fit.WorldSpace(fit.HipsFor(pose.Root));
                    body.BalanceTargetRotation = fit.Root.rotation;
                }
                if (scenario == "push" && s > 0 && s % Mathf.RoundToInt(10f / Dt) == 0)
                {
                    // A shove at the chest every 10 s, worth 0.5 m/s for her whole body.
                    var directions = new[] { Vector3.back, Vector3.forward, Vector3.left, Vector3.right, Vector3.back };
                    var chest = body.Links.FirstOrDefault(l => l.name == "Link chest") ?? body.Link(Canonical.Index("spine"));
                    chest.AddForce(directions[pushes % directions.Length] * body.Mass * 0.5f, ForceMode.Impulse);
                    pushes++;
                }
                if (scenario == "fall" && s == Mathf.RoundToInt(2f / Dt)) surfaces.Remove("window");

                body.BeforeStep(Dt);
                clock.Start();
                Physics.Simulate(Dt);
                clock.Stop();

                Trace?.Invoke(t, body, falls, surfaces, pose);
                if (falls.ImpactTime is not null && hipsAtImpact is null) hipsAtImpact = body.Links[0].transform.position;
                var head = body.Link(Canonical.Index("head")).transform.position.y;
                if (falls.Phase == FallPhase.Supported && t > 1f) lowestHead = Mathf.Min(lowestHead, head);
                foreach (var link in body.Links)
                {
                    var v = link.linearVelocity.magnitude;
                    if (v > maxSpeed) { maxSpeed = v; fastestLink = $"{link.name} at {t:0.00} s in {falls.Phase}"; }
                }
                if (falls.Phase == FallPhase.Supported && t > 2f && (scenario != "sit" || t > 5f))
                {
                    var realised = body.ReadNormalized();
                    var sum = 0f;
                    for (var i = 1; i < targets.Length; i++) sum += Quaternion.Angle(realised[i], targets[i]);
                    poseErrors.Add(sum / (targets.Length - 1));
                    if (scenario == "stand") jitter = Mathf.Max(jitter, wobble.Sample(body));
                    for (var f = 0; f < 2; f++)
                    {
                        var planted = surfaces.Contact(footColliders[f]).Gap < 0.003f && pose.Contacts[f * 2];
                        var p = feet[f].transform.position;
                        if (!planted) { plantedAt[f] = null; continue; }
                        if (plantedAt[f] is not Vector3 start) { plantedAt[f] = p; continue; }
                        footSlide = Mathf.Max(footSlide, new Vector2(p.x - start.x, p.z - start.z).magnitude);
                    }
                }
                if (scenario == "get up" && falls.Phase == FallPhase.Supported && head > 0.85f * headRest)
                {
                    if (stoodAt < 0) stoodAt = t;
                }
                else if (scenario == "get up" && stoodAt >= 0 && t - stoodAt < 3f) stoodAt = -1; // didn't stay up
                // Where her visible mesh rests, the first time she comes to rest after a fall.
                if (mesh is null && falls.Phase == FallPhase.Rested && scenario == "fall")
                {
                    body.WriteToAvatar();
                    mesh = MeshAgainstFloor(instance, 0f, true, out meshPart);
                    meshAll = MeshAgainstFloor(instance, 0f, bodyOnly: false);
                }
                if (clipping != null && s % 2 == 1)
                {
                    body.WriteToAvatar();
                    instance.Runtime.SpringBone.Process(2 * Dt);
                    clipping.Measure();
                }
                if (s >= steps - 120)
                {
                    // The last second: what the slowest-settling part is doing.
                    foreach (var link in body.Links)
                    {
                        var key = link.name;
                        var p = link.transform.position;
                        if (!lastSecondMin.ContainsKey(key)) { lastSecondMin[key] = p; lastSecondMax[key] = p; }
                        lastSecondMin[key] = Vector3.Min(lastSecondMin[key], p);
                        lastSecondMax[key] = Vector3.Max(lastSecondMax[key], p);
                    }
                }
                if (s % 1200 == 1199) yield return null;
            }

            var m = result.Metrics;
            m["falls"] = falls.Falls;
            m["physics ms per frame"] = clock.Elapsed.TotalMilliseconds / steps * 2;
            m["mean pose error deg"] = poseErrors.Count > 0 ? poseErrors.Average() : double.NaN;
            m["foot slide cm"] = footSlide * 100;
            m["fastest link m/s"] = maxSpeed;
            m["mass kg"] = body.Mass;
            if (scenario == "stand") m["wobble deg"] = jitter;
            if (scenario is "walk" or "run") m["travelled m"] = body.Links[0].transform.position.z - hipsStart.z;
            if (scenario is "walk" or "run") m["wanted m"] = (scenario == "run" ? RunSpeed * (Seconds - 0.5f) : WalkSpeed * Seconds) * fit.RootScale;
            if (mesh is { } rested)
            {
                m["mesh lowest above floor cm"] = rested.Lowest * 100;
                m["mesh deepest into floor cm"] = rested.Deepest * 100;
            }
            if (meshAll is { } all) m["mesh with hair lowest above floor cm"] = all.Lowest * 100;
            if (falls.FirstSettleSeconds is float settle) m["settle s"] = settle;
            if (!float.IsNaN(falls.FirstRestGap)) { m["rest gap cm"] = falls.FirstRestGap * 100; m["rest penetration cm"] = falls.FirstRestPenetration * 100; }
            result.Events = falls.Events.ToList();
            if (meshPart.Length > 0) result.Events.Add("mesh lowest at rest: " + meshPart);
            result.Events.Add("fastest: " + fastestLink);
            if (falls.LastMover is { } mover) result.Events.Add("last to settle: " + mover);
            if (falls.Phase == FallPhase.Settling)
            {
                var moving = body.Links.OrderByDescending(l => l.linearVelocity.magnitude).Take(3)
                    .Select(l => $"{l.name} {l.linearVelocity.magnitude:0.000} m/s, {l.angularVelocity.magnitude * Mathf.Rad2Deg:0} deg/s");
                result.Events.Add("still moving at the end: " + string.Join("; ", moving));
                result.Events.Add($"hips velocity {body.Links[0].linearVelocity}, at {body.Links[0].transform.position} (impact at {hipsAtImpact})");
                var ranges = lastSecondMax.Keys.Select(k => (k, (lastSecondMax[k] - lastSecondMin[k]).magnitude)).OrderByDescending(x => x.Item2).Take(3);
                result.Events.Add("moved in the last second: " + string.Join("; ", ranges.Select(x => $"{x.k} {x.Item2 * 100:0.0} cm")));
                var realisedNow = body.ReadNormalized();
                var off = Enumerable.Range(1, realisedNow.Length - 1).Select(i => (Canonical.Names[fit.Map.Joints[i]], Quaternion.Angle(realisedNow[i], targets[i])))
                    .OrderByDescending(x => x.Item2).Take(4);
                result.Events.Add("furthest from target: " + string.Join("; ", off.Select(x => $"{x.Item1} {x.Item2:0}°")));
                var contacts = body.Colliders.Where(c => surfaces.Contact(c).Gap < 0.005f).Select(c => c.name.Replace("Link ", "")).Distinct();
                result.Events.Add("touching the floor: " + string.Join(", ", contacts));
            }
            if (scenario == "get up") m["stood up at s"] = stoodAt;
            if (clipping != null)
            {
                m["clipping depth cm"] = clipping.MaxDepth * 100;
                m["clipping joint-frames %"] = clipping.ClippedFraction * 100;
            }
            result.HelpUsed = body.Log;
            (result.Passed, result.Reason) = Judge(scenario, result, falls, headRest, lowestHead, fit);
            if (clipping != null)
            {
                result.ClippingPassed = clipping.MaxDepth <= ClippingProbe.Allowed;
                result.ClippingNote = clipping.MaxDepth <= 0 ? $"none ({clipping.Tracked} joints tracked)"
                    : $"{clipping.MaxDepth * 100:0.0} cm ({clipping.Worst})";
            }

            clipping?.Dispose();
            body.Dispose();
            surfaces.Dispose();
            UnityEngine.Object.Destroy(instance.gameObject);
            Physics.simulationMode = previousMode;
            yield return null;
            done(result);
        }

        /// <summary>
        /// Her visible mesh against the floor (y = <paramref name="floorTop"/>): the lowest vertex's
        /// height above it (negative: sunk in) and how deep the deepest vertex goes. The fall gate
        /// is measured on the collision shapes; this checks what is actually seen.
        /// </summary>
        public static (float Lowest, float Deepest) MeshAgainstFloor(Vrm10Instance instance, float floorTop, bool bodyOnly = true) =>
            MeshAgainstFloor(instance, floorTop, bodyOnly, out _);

        public static (float Lowest, float Deepest) MeshAgainstFloor(Vrm10Instance instance, float floorTop, bool bodyOnly, out string lowestPart)
        {
            // Hair, clothing and accessories hang from spring joints; with bodyOnly, vertices that
            // mostly follow one are left out, so this measures her body.
            var springs = new HashSet<Transform>();
            if (bodyOnly)
                foreach (var spring in instance.SpringBone.Springs)
                    foreach (var joint in spring.Joints)
                        if (joint != null) springs.Add(joint.transform);
            var lowest = float.MaxValue;
            lowestPart = "";
            var baked = new Mesh();
            var vertices = new List<Vector3>();
            var weights = new List<BoneWeight>();
            foreach (var renderer in instance.GetComponentsInChildren<SkinnedMeshRenderer>())
            {
                if (!renderer.enabled || renderer.sharedMesh == null) continue;
                renderer.BakeMesh(baked, true);
                baked.GetVertices(vertices);
                renderer.sharedMesh.GetBoneWeights(weights);
                var bones = renderer.bones;
                var toWorld = Matrix4x4.TRS(renderer.transform.position, renderer.transform.rotation, Vector3.one);
                for (var i = 0; i < vertices.Count; i++)
                {
                    if (bodyOnly && i < weights.Count && weights[i].boneIndex0 < bones.Length &&
                        bones[weights[i].boneIndex0] is { } bone && springs.Contains(bone)) continue;
                    var y = toWorld.MultiplyPoint3x4(vertices[i]).y;
                    if (y < lowest)
                    {
                        lowest = y;
                        var b = i < weights.Count && weights[i].boneIndex0 < bones.Length ? bones[weights[i].boneIndex0] : null;
                        lowestPart = $"{renderer.name}/{(b != null ? b.name : "?")}";
                    }
                }
            }
            UnityEngine.Object.Destroy(baked);
            var height = lowest - floorTop;
            return (height, Mathf.Max(0f, -height));
        }

        private static (bool, string) Judge(string scenario, ScenarioResult r, FallController falls, float headRest,
            float lowestHead, AvatarFit fit)
        {
            var m = r.Metrics;
            bool Upright() => lowestHead > 0.75f * headRest;
            switch (scenario)
            {
                case "stand":
                    if (falls.Falls > 0) return (false, "fell: " + falls.Trigger);
                    if (m["mean pose error deg"] > 8) return (false, $"pose {m["mean pose error deg"]:0.0}° off target");
                    if (m["wobble deg"] > Wobble.Allowed) return (false, $"wobbles {m["wobble deg"]:0.00}° at rest");
                    return (true, "stood 60 s");
                case "walk":
                case "run":
                    if (falls.Falls > 0) return (false, "fell: " + falls.Trigger);
                    if (m["travelled m"] < 0.5 * m["wanted m"]) return (false, $"travelled {m["travelled m"]:0.0} of {m["wanted m"]:0.0} m");
                    return (true, $"{(scenario == "run" ? "ran" : "walked")} {m["travelled m"]:0.0} m");
                case "sit":
                    if (falls.Falls > 0) return (false, "fell: " + falls.Trigger);
                    var hips = r.Metrics.ContainsKey("mean pose error deg") ? m["mean pose error deg"] : 99;
                    return hips < 12 ? (true, "sat") : (false, $"pose {hips:0.0}° off the sitting pose");
                case "push":
                    return falls.Falls == 0 ? (true, "stayed up through 5 shoves") : (false, "fell: " + falls.Trigger);
                case "fall":
                    if (falls.Falls == 0) return (false, "never fell");
                    if (falls.FirstSettleSeconds is not float settle) return (false, "never came to rest: " + string.Join("; ", falls.Events));
                    if (settle > 2) return (false, $"took {settle:0.00} s to settle");
                    if (falls.FirstRestGap > 0.01f) return (false, $"resting {falls.FirstRestGap * 100:0.0} cm above the surface");
                    if (falls.FirstRestPenetration > 0.01f) return (false, $"{falls.FirstRestPenetration * 100:0.0} cm into the surface");
                    return (true, $"fell ({falls.Trigger}), settled in {settle:0.00} s");
                case "get up":
                    if (m["stood up at s"] < 0) return (false, "never stood up and stayed up 3 s");
                    return falls.Falls == 0 ? (true, $"stood up at {m["stood up at s"]:0.0} s and stayed up")
                        : (false, $"stood up, then fell again ({falls.Trigger})");
                default:
                    return (false, "unknown scenario");
            }
        }
    }

    /// <summary>
    /// Jitter as visible wobble: how far each link turns from where it was a second ago, in
    /// degrees, while standing still (a breathing sway is part of the pose, so it's compared
    /// with the target, link by link).
    /// </summary>
    public sealed class Wobble
    {
        public const float Allowed = 0.5f;
        private readonly Queue<Quaternion[]> _history = new();

        public float Sample(PhysicsBody body)
        {
            var now = body.Links.Select(l => l.transform.rotation).ToArray();
            _history.Enqueue(now);
            if (_history.Count < 13) return 0;
            // Compare with 1/10 s ago (12 steps): fast back-and-forth shows up, slow sway doesn't.
            var then = _history.Dequeue();
            var worst = 0f;
            for (var i = 0; i < now.Length; i++) worst = Mathf.Max(worst, Quaternion.Angle(now[i], then[i]));
            return worst;
        }
    }

    /// <summary>
    /// Roadmap A1's clipping check: does a hair, clothing or accessory joint pass into the body?
    /// Each chain is tested against the body colliders it is meant to collide with (SORA's
    /// generated ones and any the avatar ships with), by the joint's centre. Chains with no body
    /// collision (left out because they rest against the body) aren't tested.
    /// </summary>
    public sealed class ClippingProbe : IDisposable
    {
        public const float Allowed = 0.01f;

        private readonly List<(Transform Joint, VRM10SpringBoneCollider[] Shapes)> _joints = new();
        private long _samples, _clipped;

        public float MaxDepth { get; private set; }
        public string Worst { get; private set; } = "";
        public double ClippedFraction => _samples == 0 ? 0 : (double)_clipped / _samples;
        public int Tracked => _joints.Count;

        public ClippingProbe(Vrm10Instance instance, PhysicsBody body)
        {
            foreach (var spring in instance.SpringBone.Springs)
            {
                var shapes = spring.ColliderGroups.Where(g => g != null).SelectMany(g => g.Colliders)
                    .Where(c => c != null && c.ColliderType is VRM10SpringBoneColliderTypes.Sphere or VRM10SpringBoneColliderTypes.Capsule)
                    .ToArray();
                if (shapes.Length == 0) continue;
                for (var j = 1; j < spring.Joints.Count; j++)
                    if (spring.Joints[j] != null && Depth(spring.Joints[j].transform.position, shapes, out _) <= 0)
                        _joints.Add((spring.Joints[j].transform, shapes));
            }
        }

        public void Measure()
        {
            foreach (var (joint, shapes) in _joints)
            {
                _samples++;
                var depth = Depth(joint.position, shapes, out var shape);
                if (depth <= 0.002f) continue;
                _clipped++;
                if (depth > MaxDepth) { MaxDepth = depth; Worst = joint.name + " in " + shape; }
            }
        }

        private static float Depth(Vector3 p, VRM10SpringBoneCollider[] shapes, out string deepest)
        {
            var worst = 0f; deepest = "";
            foreach (var c in shapes)
            {
                var t = c.transform;
                var a = t.TransformPoint(c.Offset);
                var radius = c.Radius * t.lossyScale.x;
                var closest = a;
                if (c.ColliderType == VRM10SpringBoneColliderTypes.Capsule)
                {
                    var b = t.TransformPoint(c.Tail);
                    var ab = b - a;
                    var s = ab.sqrMagnitude > 1e-10f ? Mathf.Clamp01(Vector3.Dot(p - a, ab) / ab.sqrMagnitude) : 0;
                    closest = a + s * ab;
                }
                var depth = radius - Vector3.Distance(p, closest);
                if (depth > worst) { worst = depth; deepest = t.name; }
            }
            return worst;
        }

        public void Dispose() { }
    }
}
