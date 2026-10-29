using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using MotionEnv.AvatarLayer;
using MotionEnv.Environment;
using NUnit.Framework;
using UniVRM10;
using UnityEngine;
using UnityEngine.TestTools;
using Debug = UnityEngine.Debug;

/// <summary>Roadmap MA, A3: the physics body.</summary>
public sealed class A3PhysicsBodyTests
{
    public const float Dt = 1f / 120f;

    [SetUp]
    public void ScriptedPhysics()
    {
        Physics.simulationMode = SimulationMode.Script;
        Physics.gravity = new Vector3(0, -9.81f, 0);
    }

    [TearDown]
    public void AutomaticPhysics() => Physics.simulationMode = SimulationMode.FixedUpdate;

    /// <summary>
    /// Unity doesn't document its drive units. A 2 kg weight on a 0.5 m arm (9.81 N·m) hangs
    /// from a motor of known stiffness; how far it sags says whether stiffness is per radian or
    /// per degree.
    /// </summary>
    [Test]
    public void DriveUnitsAreCalibrated()
    {
        var root = new GameObject("calibration").AddComponent<ArticulationBody>();
        root.immovable = true;
        var arm = new GameObject("arm");
        arm.transform.SetParent(root.transform, false);
        var link = arm.AddComponent<ArticulationBody>();
        link.jointType = ArticulationJointType.SphericalJoint;
        link.twistLock = link.swingYLock = link.swingZLock = ArticulationDofLock.LimitedMotion;
        const float k = 100f, mass = 2f, lever = 0.5f;
        ArticulationDrive D() => new() { lowerLimit = -170, upperLimit = 170, stiffness = k, damping = 20, forceLimit = 1e6f, driveType = ArticulationDriveType.Force };
        link.xDrive = D(); link.yDrive = D(); link.zDrive = D();
        link.mass = mass;
        link.automaticCenterOfMass = false;
        link.centerOfMass = new Vector3(lever, 0, 0);
        link.automaticInertiaTensor = false;
        link.inertiaTensor = new Vector3(0.01f, 0.01f, 0.01f);
        link.inertiaTensorRotation = Quaternion.identity;
        link.linearDamping = link.angularDamping = 0.5f;
        for (var i = 0; i < 1200; i++) Physics.Simulate(Dt);
        var sag = Quaternion.Angle(Quaternion.identity, arm.transform.localRotation);
        var torque = mass * 9.81f * lever;
        var perRadian = torque / k * Mathf.Rad2Deg;   // degrees, if k is N·m/rad
        var perDegree = torque / k;                   // degrees, if k is N·m/deg
        Debug.Log($"A3 calibration: sag {sag:0.00}°; per radian predicts {perRadian:0.00}°, per degree predicts {perDegree:0.00}°");
        // Joint friction makes the sag a little smaller than the prediction.
        var unitsPerSi = Mathf.Abs(sag - perDegree) < Mathf.Abs(sag - perRadian) ? Mathf.Deg2Rad : 1f;
        Assert.That(sag, Is.InRange(0.6f * perRadian, 1.1f * perRadian), "stiffness is per radian");
        Assert.AreEqual(unitsPerSi, PhysicsBody.DriveUnitsPerSi, 1e-6f,
            "PhysicsBody.DriveUnitsPerSi must match the measurement");
        Object.Destroy(root.gameObject);
    }

    public static IEnumerator Build(string avatar, System.Action<Vrm10Instance, PhysicsBody, DesktopSurfaces> done)
    {
        Vrm10Instance instance = null!;
        yield return TestAvatars.Load(avatar, i => instance = i);
        var fit = new AvatarFit(instance.GetComponent<Animator>());
        var body = new PhysicsBody(fit, groundY: 0);
        var floor = TestAvatars.Floor();
        Physics.SyncTransforms();
        done(instance, body, floor);
    }

    private static float PoseError(PhysicsBody body, Quaternion[] target, out string worstJoint)
    {
        var realised = body.ReadNormalized();
        var worst = 0f; worstJoint = "";
        var off = new List<string>();
        for (var i = 1; i < target.Length; i++)
        {
            var e = Quaternion.Angle(realised[i], target[i]);
            if (e > 4) off.Add($"{Canonical.Names[body.Fit.Map.Joints[i]]} {e:0}°");
            if (e > worst) { worst = e; worstJoint = Canonical.Names[body.Fit.Map.Joints[i]]; }
        }
        if (off.Count > 0) worstJoint += " (off: " + string.Join(", ", off) + ")";
        return worst;
    }

    /// <summary>Without gravity, with strong motors, every joint reaches its target pose: this
    /// checks the rotation-vector drive targets and the joint limits' signs.</summary>
    [UnityTest]
    public IEnumerator MotorsReachTheirTargets()
    {
        var avatar = TestAvatars.Available("Alice") ? "Alice" : "Seed-san";
        Vrm10Instance instance = null!; PhysicsBody body = null!; DesktopSurfaces floor = null!;
        yield return Build(avatar, (i, b, f) => { instance = i; body = b; floor = f; });
        Physics.gravity = Vector3.zero;
        body.Help = new HelpDial(0, 1, 1);
        var motion = new StandInMotion();
        var poses = new List<(string, CanonicalPose)>();
        void Add(string name, System.Action<CanonicalPose> make) { var p = new CanonicalPose(); make(p); poses.Add((name, p)); }
        Add("stand", p => motion.Stand(0, p));
        Add("sit", p => motion.Sit(1, p));
        Add("kneel", motion.Kneel);
        Add("fall forward", p => motion.FallReaction(new Vector2(0, 1), p));
        Add("fall right", p => motion.FallReaction(new Vector2(1, 0), p));
        foreach (var (name, pose) in poses)
        {
            var target = body.Fit.Fold(pose);
            body.SetTargets(target);
            body.BalanceTargetPosition = body.Links[0].transform.position;
            for (var i = 0; i < 360; i++) { body.BeforeStep(Dt); Physics.Simulate(Dt); }
            var error = PoseError(body, target, out var joint);
            Debug.Log($"A3 drive accuracy [{avatar}, {name}]: worst joint {joint} {error:0.0}°");
            Assert.Less(error, 4f, $"{name}: {joint}");
        }
        body.Dispose(); floor.Dispose(); Object.Destroy(instance.gameObject);
    }

    /// <summary>The balance helper actually pulls: with no gravity, a target 0.3 m away and
    /// tipped 30° is reached, in the default physics scene and in a scene of its own (as in
    /// SORA's player).</summary>
    [UnityTest]
    public IEnumerator TheBalanceHelperPullsPositionAndRotation()
    {
        foreach (var ownScene in new[] { false, true })
        {
            Vrm10Instance instance = null!;
            yield return TestAvatars.Load("Seed-san", i => instance = i);
            var fit = new AvatarFit(instance.GetComponent<Animator>());
            UnityEngine.SceneManagement.Scene scene = default;
            var previous = UnityEngine.SceneManagement.SceneManager.GetActiveScene();
            if (ownScene)
            {
                scene = UnityEngine.SceneManagement.SceneManager.CreateScene("balance test",
                    new UnityEngine.SceneManagement.CreateSceneParameters(UnityEngine.SceneManagement.LocalPhysicsMode.Physics3D));
                UnityEngine.SceneManagement.SceneManager.SetActiveScene(scene);
            }
            var body = new PhysicsBody(fit, origin: new Vector3(0, 5, 0));
            UnityEngine.SceneManagement.SceneManager.SetActiveScene(previous);
            var physics = ownScene ? scene.GetPhysicsScene() : Physics.defaultPhysicsScene;
            Physics.gravity = Vector3.zero;
            body.Help = new HelpDial(0, 0, 1);
            var start = body.Links[0].transform.position;
            body.BalanceTargetPosition = start + new Vector3(0.3f, 0, 0);
            body.BalanceTargetRotation = Quaternion.Euler(30, 0, 0);
            for (var i = 0; i < 240; i++) { body.BeforeStep(Dt); physics.Simulate(Dt); }
            var moved = body.Links[0].transform.position.x - start.x;
            var turned = Quaternion.Angle(Quaternion.identity, body.Links[0].transform.rotation);
            Debug.Log($"A3 balance helper ({(ownScene ? "own scene" : "default scene")}): moved {moved:0.000} of 0.3 m, turned {turned:0.0} of 30°");
            Assert.AreEqual(0.3f, moved, 0.03f, "position pull");
            Assert.AreEqual(30f, turned, 3f, "rotation pull");
            Debug.Log($"A3 balance helper logged {body.Log.BalanceImpulse:0.0} N·s and {body.Log.BalanceAngularImpulse:0.0} N·m·s, peak {body.Log.PeakBalanceForce:0} N");
            Assert.Greater(body.Log.BalanceImpulse, 0.1, "its use is logged");
            Assert.Greater(body.Log.BalanceAngularImpulse, 0.1, "its turning is logged");
            body.Dispose(); Object.Destroy(instance.gameObject);
            if (ownScene) UnityEngine.SceneManagement.SceneManager.UnloadSceneAsync(scene);
            Physics.gravity = new Vector3(0, -9.81f, 0);
            yield return null;
        }
    }

    /// <summary>SORA draws her scaled down (1 unit tall). The body simulates in her own metres, so
    /// its collision shapes, mesh-fitted included, must be the same at any display scale.</summary>
    [UnityTest]
    public IEnumerator CollisionShapesAreTheSameAtAnyDisplayScale()
    {
        var sizes = new System.Collections.Generic.List<float[]>();
        foreach (var scale in new[] { 1f, 0.456f })
        {
            Vrm10Instance instance = null!;
            yield return TestAvatars.Load("Seed-san", i => instance = i);
            instance.transform.localScale = Vector3.one * scale;
            var body = new PhysicsBody(new AvatarFit(instance.GetComponent<Animator>()), origin: new Vector3(0, 10, 0));
            Physics.SyncTransforms();
            sizes.Add(body.Colliders.Select(c => c.bounds.size.magnitude).ToArray());
            body.Dispose(); Object.Destroy(instance.gameObject);
            yield return null;
        }
        var worst = sizes[0].Zip(sizes[1], (a, b) => Mathf.Abs(a - b) / Mathf.Max(a, 1e-4f)).Max();
        Debug.Log($"A3 display scale: collision shapes differ by at most {worst * 100:0.00}%");
        Assert.Less(worst, 0.01f);
    }

    /// <summary>Held at a point (as when grabbed), she hangs from it and swings freely.</summary>
    [UnityTest]
    public IEnumerator HeldByTheHeadSheHangsBelowIt()
    {
        Vrm10Instance instance = null!; PhysicsBody body = null!; DesktopSurfaces floor = null!;
        yield return Build("Seed-san", (i, b, f) => { instance = i; body = b; floor = f; });
        floor.Dispose();
        body.Help = new HelpDial(0, 0, 1);
        body.BalanceStrength = 4f;
        var head = body.Link(Canonical.Index("head"));
        var grip = head.transform.position + Vector3.up * 0.05f;
        body.HoldAt(head, grip, holdRotation: false);
        for (var i = 0; i < 480; i++) { body.BalanceTargetPosition = grip; body.BeforeStep(Dt); Physics.Simulate(Dt); }
        var heldGap = Vector3.Distance(body.HeldPoint, grip);
        var hipsBelow = grip.y - body.Links[0].transform.position.y;
        Debug.Log($"A3 held by the head: grip {heldGap * 100:0.0} cm from the hand, hips {hipsBelow:0.00} m below it");
        Assert.Less(heldGap, 0.05f, "the grip holds");
        Assert.Greater(hipsBelow, 0.4f, "she hangs below the grip");
        body.HoldAtHips();
        Assert.AreSame(body.Links[0], body.HeldLink);
        body.Dispose(); Object.Destroy(instance.gameObject);
    }

    /// <summary>The steadiness and cost gates: standing with full help, the pose stays within a
    /// few degrees of its target with no jitter, and physics at 120 Hz costs under 1 ms a frame.</summary>
    [UnityTest]
    public IEnumerator StandingIsSteadyAndCheap()
    {
        foreach (var avatar in TestAvatars.All)
        {
            if (!TestAvatars.Available(avatar)) continue;
            Vrm10Instance instance = null!; PhysicsBody body = null!; DesktopSurfaces floor = null!;
            yield return Build(avatar, (i, b, f) => { instance = i; body = b; floor = f; });
            body.Help = HelpDial.Ladder[0].Dial;
            var pose = new CanonicalPose();
            new StandInMotion().Stand(0, pose);
            var target = body.Fit.Fold(pose);
            body.Teleport(body.Links[0].transform.position, body.Links[0].transform.rotation, target);
            body.BalanceTargetPosition = body.Fit.WorldSpace(body.Fit.HipsFor(pose.Root));
            for (var i = 0; i < 240; i++) { body.BeforeStep(Dt); Physics.Simulate(Dt); }
            var clock = new Stopwatch();
            var worstWobble = 0f;
            var jitter = 0f; var fastest = "";
            var wobble = new Wobble();
            for (var i = 0; i < 240; i++)
            {
                body.BeforeStep(Dt);
                clock.Start(); Physics.Simulate(Dt); clock.Stop();
                foreach (var link in body.Links) { var w = link.angularVelocity.magnitude * Mathf.Rad2Deg; if (w > jitter) { jitter = w; fastest = link.name; } }
                worstWobble = Mathf.Max(worstWobble, wobble.Sample(body));
            }
            var error = PoseError(body, target, out var joint);
            var perFrameMs = clock.Elapsed.TotalMilliseconds / 240 * 2; // two 120 Hz steps per 60 Hz frame
            Debug.Log($"A3 steadiness [{avatar}]: worst joint {joint} {error:0.0}° off target, fastest link {fastest} {jitter:0.0}°/s, wobble {worstWobble:0.000}° per 0.1 s, " +
                      $"physics {perFrameMs:0.000} ms a frame, mass {body.Mass:0.0} kg");
            Assert.Less(error, 5f, joint);
            Assert.Less(worstWobble, Wobble.Allowed, "no visible jitter at rest");
            Assert.Less(perFrameMs, 1.0, "physics at 120 Hz costs at most 1 ms a frame");
            body.Dispose(); floor.Dispose(); Object.Destroy(instance.gameObject);
            yield return null;
        }
    }
}
