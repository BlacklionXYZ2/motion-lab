using System;
using System.Collections.Generic;
using System.Linq;
using UniVRM10;
using UnityEngine;

namespace MotionEnv.AvatarLayer
{
    /// <summary>
    /// A3, the physics body: one articulation link per avatar joint, built from the avatar's own
    /// bones, with masses from standard body-segment proportions, anatomical joint limits, and
    /// joint motors (PD drives) pulling towards the target pose with realistic torque limits.
    /// Help comes only from the <see cref="HelpDial"/>, and every use is logged.
    /// <para>It simulates in metres at the avatar's own size: real gravity, real time. Mapping onto
    /// screen pixels is the display's job. The links live in their own hierarchy; the avatar's
    /// bones are only written from them (<see cref="WriteToAvatar"/>).</para>
    /// </summary>
    public sealed class PhysicsBody : IDisposable
    {
        /// <summary>Unity's drive stiffness and damping units per SI unit (N·m/rad, N·m·s/rad).
        /// Measured by the calibration test.</summary>
        public static float DriveUnitsPerSi = 1f;

        /// <summary>Smallest rotational inertia of any link, kg·m², for a 45 kg body.</summary>
        public const float MinimumInertia = 0.004f;

        /// <summary>Error at which a motor reaches its torque limit (radians).</summary>
        public const float SaturationRadians = 0.35f;
        /// <summary>Damping as a time constant (seconds) times stiffness.</summary>
        public const float DampingSeconds = 0.06f;
        /// <summary>Passive damping never falls below that of this muscle tone.</summary>
        public const float PassiveDampingTone = 0.45f;
        /// <summary>Help at full strength multiplies motors by this.</summary>
        public const float StrengthHelpFactor = 10f;
        /// <summary>Exempt joints at full help: stiffness multiplier, with no torque limit.</summary>
        public const float ExemptStiffnessFactor = 40f;

        private readonly AvatarFit _fit;
        private readonly Quaternion _frame;            // the avatar root's rotation (world-aligned rest frames)
        private readonly ArticulationBody[] _links;    // per avatar joint
        private readonly float[] _realisticTorque;     // N·m
        private readonly float[] _tone;                // per avatar joint, 0..1 (A4 lowers it)
        private readonly Quaternion[] _normalized;
        private HelpDial _appliedHelp = new(-1, -1, -1);

        public GameObject Container { get; }
        public IReadOnlyList<ArticulationBody> Links => _links;
        public IReadOnlyList<Collider> Colliders { get; }
        public float Mass { get; }
        public float Height => _fit.Height;
        public HelpLog Log { get; } = new();
        public HelpDial Help { get; set; }
        /// <summary>Where the balance helper pulls the hips (world) and how it turns them.</summary>
        public Vector3 BalanceTargetPosition { get; set; }
        public Quaternion BalanceTargetRotation { get; set; }
        public AvatarFit Fit => _fit;
        /// <summary>Where the avatar's root was in simulation space when the body was built.</summary>
        public Vector3 Origin { get; }
        /// <summary>How far the avatar's root has moved since (simulation space); a display that
        /// moves the avatar keeps this up to date.</summary>
        public Vector3 RootOffset { get; set; }
        /// <summary>The simulation frame's rotation (the avatar's facing).</summary>
        public Quaternion Frame => _frame;

        /// <summary>A point in the avatar's root space (metres) in simulation space.</summary>
        public Vector3 SimPoint(Vector3 rootSpace) => Origin + RootOffset + _frame * rootSpace;

        private readonly float _ground;

        /// <param name="groundY">The floor's height under her rest pose (simulation space); the origin's by default.</param>
        /// <param name="origin">Where the avatar's root is in simulation space. Omitted, the body
        /// simulates where the avatar stands (which must then be at scale 1). Given, it simulates
        /// there in metres, facing +Z, whatever the avatar's display scale: a display maps it back.</param>
        /// <param name="fitToMesh">Size each collision shape from the avatar's own mesh (the
        /// default), so what she rests on is what is seen; otherwise from her height alone.</param>
        public PhysicsBody(AvatarFit fit, float? massKg = null, float? groundY = null, Vector3? origin = null, bool fitToMesh = true)
        {
            _fit = fit ?? throw new ArgumentNullException(nameof(fit));
            if (origin is null && Mathf.Abs(fit.Root.lossyScale.y - 1f) > 1e-3f)
                throw new ArgumentException("simulating where the avatar stands needs scale 1; pass an origin instead");
            _frame = origin is null ? fit.Root.rotation : Quaternion.identity;
            Origin = origin ?? fit.Root.position;
            _ground = groundY ?? Origin.y;
            var n = fit.Map.Joints.Count;
            _links = new ArticulationBody[n];
            _realisticTorque = new float[n];
            _tone = Enumerable.Repeat(1f, n).ToArray();
            _normalized = new Quaternion[n];
            var height = fit.Height;
            Mass = massKg ?? 21f * height * height; // a slim adult build (BMI 21)
            Container = new GameObject("Physics body");
            Container.transform.SetPositionAndRotation(Origin, _frame);

            var rest = Enumerable.Range(0, n).Select(i => SimPoint(fit.RestPosition(i))).ToArray();
            _meshPoints = new List<Vector3>?[n];
            _accessoryPoints = new List<List<Vector3>>?[n];
            if (fitToMesh) MeshPointsByLink();
            var masses = SegmentMasses(fit.Map);
            var colliders = new List<Collider>();
            for (var i = 0; i < n; i++)
            {
                var joint = fit.Map.Joints[i];
                var parent = fit.Map.PresentParent(joint);
                var parentIndex = parent < 0 ? -1 : fit.IndexOf(parent);
                var go = new GameObject("Link " + Canonical.Names[joint]);
                go.transform.SetParent(parentIndex < 0 ? Container.transform : _links[parentIndex].transform, false);
                go.transform.SetPositionAndRotation(rest[i], _frame);
                var link = go.AddComponent<ArticulationBody>();
                _links[i] = link;
                link.mass = masses[i] * Mass;
                link.linearDamping = 0.02f;
                link.angularDamping = 0.05f;
                link.jointFriction = 0.02f;
                link.maxDepenetrationVelocity = 1.5f;
                if (parentIndex < 0)
                {
                    link.immovable = false;
                }
                else
                {
                    link.jointType = ArticulationJointType.SphericalJoint;
                    link.matchAnchors = true;
                    link.anchorPosition = Vector3.zero;
                    link.anchorRotation = Quaternion.identity;
                    link.twistLock = ArticulationDofLock.LimitedMotion;
                    link.swingYLock = ArticulationDofLock.LimitedMotion;
                    link.swingZLock = ArticulationDofLock.LimitedMotion;
                    var (lx, ly, lz) = Limits(joint);
                    link.xDrive = Drive(lx);
                    link.yDrive = Drive(ly);
                    link.zDrive = Drive(lz);
                }
                _realisticTorque[i] = TorquePerKg(joint) * Mass;
                foreach (var c in BuildColliders(go, i, joint, rest, height))
                {
                    c.sharedMaterial = SkinMaterial;
                    colliders.Add(c);
                }
            }
            Colliders = colliders;
            IgnoreSelfOverlaps(colliders);
            foreach (var link in _links) link.automaticCenterOfMass = link.automaticInertiaTensor = true;
            Physics.SyncTransforms();
            // Light links carrying heavy children (a collarbone holding an arm, a hand) make PhysX
            // articulations unstable; a floor on rotational inertia keeps them steady.
            foreach (var link in _links)
            {
                var inertia = link.inertiaTensor;
                var floor = MinimumInertia * Mass / 45f;
                if (inertia.x >= floor && inertia.y >= floor && inertia.z >= floor) continue;
                var rotation = link.inertiaTensorRotation;
                link.automaticInertiaTensor = false;
                link.inertiaTensor = new Vector3(Mathf.Max(inertia.x, floor), Mathf.Max(inertia.y, floor), Mathf.Max(inertia.z, floor));
                link.inertiaTensorRotation = rotation;
            }
            Help = HelpDial.None;
            BalanceTargetPosition = _links[0].transform.position;
            BalanceTargetRotation = _frame;
            BuildBalanceAnchor();
            ApplyMotorSettings(force: true);
        }

        public ArticulationBody Link(int canonical)
        {
            var i = _fit.IndexOf(canonical);
            return i < 0 ? throw new ArgumentException("the avatar lacks " + Canonical.Names[canonical]) : _links[i];
        }

        public Vector3 CenterOfMass
        {
            get
            {
                var sum = Vector3.zero;
                foreach (var link in _links) sum += link.worldCenterOfMass * link.mass;
                return sum / Mass;
            }
        }

        /// <summary>Angular damping on every link (null: the normal 0.05, air only).</summary>
        public void SetAngularDamping(float? damping)
        {
            foreach (var link in _links) link.angularDamping = damping ?? 0.05f;
        }

        /// <summary>Muscle tone per joint, 0 to 1 (falling lowers it; the dial's help is on top).</summary>
        public void SetTone(Func<int, float> toneForCanonical)
        {
            for (var i = 0; i < _links.Length; i++) _tone[i] = Mathf.Clamp01(toneForCanonical(_fit.Map.Joints[i]));
            ApplyMotorSettings(force: true);
        }

        /// <summary>Sets every motor's target from normalized local rotations (per avatar joint).</summary>
        public void SetTargets(Quaternion[] normalizedLocal)
        {
            for (var i = 1; i < _links.Length; i++)
            {
                var v = RotationVectorDegrees(normalizedLocal[i]);
                var link = _links[i];
                var x = link.xDrive; x.target = v.x; link.xDrive = x;
                var y = link.yDrive; y.target = v.y; link.yDrive = y;
                var z = link.zDrive; z.target = v.z; link.zDrive = z;
            }
        }

        /// <summary>Call once per physics step, before <c>Physics.Simulate</c>.</summary>
        public void BeforeStep(float dt)
        {
            ApplyMotorSettings(force: false);
            if (Help.Exempt > 0) Log.ExemptSeconds += dt;
            if (Help.Strength > 0) Log.StrengthSeconds += dt;
            // The balance helper is a joint to an invisible anchor at the target, solved implicitly
            // by PhysX, so it is stable at any strength. Its force is logged as help.
            if (_anchorBalance != Help.Balance || _anchorExempt != Help.Exempt || _anchorVertical != BalanceVerticalShare || _anchorStrength != BalanceStrength)
                ConfigureBalance();
            var held = HeldPoint;
            if (Help.Balance <= 0) { _lastHeld = held; return; }
            _anchor.MovePosition(BalanceTargetPosition);
            _anchor.MoveRotation(BalanceTargetRotation);
            // Unity reports no force for a joint attached to an articulation, so the log uses the
            // drives' own spring law on this step's error (what PhysX applies, before its solve).
            var velocity = _lastHeld is Vector3 last ? (held - last) / dt : Vector3.zero;
            _lastHeld = held;
            var error = BalanceTargetPosition - held;
            var x = _balanceJoint.xDrive;
            var y = _balanceJoint.yDrive;
            static float Axis(JointDrive d, float e, float v) =>
                Mathf.Clamp(d.positionSpring * e - d.positionDamper * v, -d.maximumForce, d.maximumForce);
            var force = new Vector3(Axis(x, error.x, velocity.x), Axis(y, error.y, velocity.y), Axis(x, error.z, velocity.z)).magnitude;
            var torque = 0f;
            if (_holdRotation)
            {
                (BalanceTargetRotation * Quaternion.Inverse(_heldLink.transform.rotation)).ToAngleAxis(out var angle, out _);
                if (angle > 180) angle -= 360;
                var s = _balanceJoint.slerpDrive;
                torque = Mathf.Min(s.positionSpring * Mathf.Abs(angle) * Mathf.Deg2Rad, s.maximumForce);
            }
            Log.BalanceImpulse += force * dt;
            Log.BalanceAngularImpulse += torque * dt;
            Log.PeakBalanceForce = Math.Max(Log.PeakBalanceForce, force);
        }

        private Vector3? _lastHeld;
        private ArticulationBody _heldLink = null!;
        private Vector3 _heldLocal;
        private bool _holdRotation = true;

        /// <summary>The point the balance helper holds (world); the hips unless she is held elsewhere.</summary>
        public Vector3 HeldPoint => _heldLink.transform.TransformPoint(_heldLocal);
        /// <summary>The link the balance helper holds.</summary>
        public ArticulationBody HeldLink => _heldLink;

        /// <summary>Multiplies the balance helper's strength (1 by default). A hand holding her
        /// isn't a muscle: it can pull harder than the helper normally may.</summary>
        public float BalanceStrength { get; set; } = 1f;

        /// <summary>
        /// Holds her at a point (world) on one link instead of at the hips, e.g. where she was
        /// grabbed. Without the rotation hold she swings freely from it.
        /// </summary>
        public void HoldAt(ArticulationBody link, Vector3 worldPoint, bool holdRotation)
        {
            _heldLink = link;
            _heldLocal = link.transform.InverseTransformPoint(worldPoint);
            _holdRotation = holdRotation;
            _lastHeld = null;
            // The joint's reference is the relative pose at connection: line the anchor up first.
            _anchor.position = worldPoint;
            _anchor.rotation = link.transform.rotation;
            _balanceJoint.connectedArticulationBody = link;
            _balanceJoint.anchor = Vector3.zero;
            _balanceJoint.connectedAnchor = _heldLocal;
            BalanceTargetPosition = worldPoint;
            BalanceTargetRotation = link.transform.rotation;
            ConfigureBalance();
        }

        /// <summary>Back to holding her at the hips, upright.</summary>
        public void HoldAtHips() => HoldAt(_links[0], _links[0].transform.position, true);

        private Rigidbody _anchor = null!;
        private ConfigurableJoint _balanceJoint = null!;
        /// <summary>The balance helper's anchor and joint (for inspection).</summary>
        public Rigidbody BalanceAnchor => _anchor;
        public ConfigurableJoint BalanceJoint => _balanceJoint;
        private float _anchorBalance = -1, _anchorExempt = -1, _anchorVertical = -1, _anchorStrength = -1;

        /// <summary>How much of the balance helper's pull is vertical (0 to 1; 1 by default).
        /// Below 1, her legs carry her weight and the helper mostly keeps her upright and over
        /// her feet, so a stiff height target can't tip her pelvis when her feet are apart.</summary>
        public float BalanceVerticalShare { get; set; } = 1f;

        private void BuildBalanceAnchor()
        {
            var go = new GameObject("Balance anchor");
            go.transform.SetParent(Container.transform, false);
            go.transform.SetPositionAndRotation(_links[0].transform.position, _links[0].transform.rotation);
            _anchor = go.AddComponent<Rigidbody>();
            _anchor.isKinematic = true;
            _anchor.interpolation = RigidbodyInterpolation.None;
            _balanceJoint = go.AddComponent<ConfigurableJoint>();
            _balanceJoint.connectedArticulationBody = _links[0];
            _balanceJoint.autoConfigureConnectedAnchor = false;
            _balanceJoint.anchor = Vector3.zero;
            _balanceJoint.connectedAnchor = Vector3.zero;
            _balanceJoint.xMotion = _balanceJoint.yMotion = _balanceJoint.zMotion = ConfigurableJointMotion.Free;
            _balanceJoint.angularXMotion = _balanceJoint.angularYMotion = _balanceJoint.angularZMotion = ConfigurableJointMotion.Free;
            _balanceJoint.rotationDriveMode = RotationDriveMode.Slerp;
            _balanceJoint.enableCollision = false;
            _heldLink = _links[0];
            _heldLocal = Vector3.zero;
            ConfigureBalance();
        }

        private void ConfigureBalance()
        {
            _anchorBalance = Help.Balance;
            _anchorExempt = Help.Exempt;
            _anchorVertical = BalanceVerticalShare;
            _anchorStrength = BalanceStrength;
            var b = Help.Balance * BalanceStrength;
            // Whole-body natural frequency about 20 rad/s for position, near critical damping;
            // at most four times her weight.
            var linear = new JointDrive { positionSpring = b * Mass * 400f, positionDamper = b * Mass * 36f, maximumForce = b * Mass * 9.81f * 4f };
            _balanceJoint.xDrive = _balanceJoint.zDrive = linear;
            var v = Mathf.Clamp01(BalanceVerticalShare);
            _balanceJoint.yDrive = new JointDrive { positionSpring = linear.positionSpring * v,
                positionDamper = linear.positionDamper * Mathf.Max(v, 0.3f), maximumForce = linear.maximumForce * v };
            // Exempt joints make her limbs nearly rigid, so a foot meeting the floor early can lever
            // her whole body; the rotation hold grows with them so the hips stay the anchor.
            var rigid = Mathf.Lerp(1f, ExemptStiffnessFactor, Help.Exempt);
            _balanceJoint.slerpDrive = _holdRotation
                ? new JointDrive { positionSpring = b * Mass * 60f * rigid, positionDamper = b * Mass * 6f * Mathf.Sqrt(rigid), maximumForce = b * Mass * 30f * rigid }
                : new JointDrive { positionSpring = 0f, positionDamper = b * Mass * 0.2f, maximumForce = b * Mass * 30f };
        }

        /// <summary>Reads the links back as normalized local rotations (per avatar joint).</summary>
        public Quaternion[] ReadNormalized()
        {
            var inverseFrame = Quaternion.Inverse(_frame);
            for (var i = 0; i < _links.Length; i++)
            {
                var world = inverseFrame * _links[i].transform.rotation;
                var parent = _fit.Map.PresentParent(_fit.Map.Joints[i]);
                _normalized[i] = parent < 0 ? world
                    : Quaternion.Inverse(inverseFrame * _links[_fit.IndexOf(parent)].transform.rotation) * world;
            }
            return _normalized;
        }

        /// <summary>Poses the avatar's bones from the links. <paramref name="displayTurn"/> is how
        /// far the display has turned the avatar since the body was built (about the vertical);
        /// the body turns physically, so it is taken out again.</summary>
        /// <param name="displayRoot">Where the avatar's root is now, in simulation space
        /// (<c>Origin + RootOffset</c> by default). Passing the display's current root keeps the
        /// body where it is in the world even if the display moved its root meanwhile.</param>
        public void WriteToAvatar(Quaternion? displayTurn = null, Vector3? displayRoot = null)
        {
            var undo = Quaternion.Inverse(displayTurn ?? Quaternion.identity);
            var normalized = ReadNormalized();
            normalized[0] = undo * normalized[0];
            var root = displayRoot ?? Origin + RootOffset;
            _fit.ApplyNormalized(normalized, undo * (Quaternion.Inverse(_frame) * (_links[0].transform.position - root)));
        }

        /// <summary>Puts the whole body at a pose instantly (no velocities), e.g. to start a scenario.</summary>
        public void Teleport(Vector3 hipsWorld, Quaternion hipsWorldRotation, Quaternion[] normalizedLocal)
        {
            _links[0].TeleportRoot(hipsWorld, hipsWorldRotation);
            if (_heldLink != _links[0]) HoldAt(_links[0], _links[0].transform.position, true);
            _anchor.position = hipsWorld; _anchor.rotation = hipsWorldRotation;
            BalanceTargetPosition = hipsWorld; BalanceTargetRotation = hipsWorldRotation;
            var positions = new List<float>();
            _links[0].GetJointPositions(positions);
            var starts = new List<int>();
            _links[0].GetDofStartIndices(starts);
            for (var i = 1; i < _links.Length; i++)
            {
                var start = starts[_links[i].index];
                var v = RotationVectorDegrees(normalizedLocal[i]) * Mathf.Deg2Rad;
                if (start + 2 < positions.Count)
                {
                    positions[start] = v.x; positions[start + 1] = v.y; positions[start + 2] = v.z;
                }
            }
            _links[0].SetJointPositions(positions);
            var zeros = Enumerable.Repeat(0f, positions.Count).ToList();
            _links[0].SetJointVelocities(zeros);
            _links[0].linearVelocity = Vector3.zero;
            _links[0].angularVelocity = Vector3.zero;
            SetTargets(normalizedLocal);
        }

        /// <summary>Exponential map: a rotation as axis × angle, in degrees per axis.</summary>
        public static Vector3 RotationVectorDegrees(Quaternion q)
        {
            if (q.w < 0) q = new Quaternion(-q.x, -q.y, -q.z, -q.w);
            q.ToAngleAxis(out var angle, out var axis);
            if (!float.IsFinite(axis.x) || angle < 1e-4f) return Vector3.zero;
            return axis.normalized * angle;
        }

        private void ApplyMotorSettings(bool force)
        {
            if (!force && Help.Exempt == _appliedHelp.Exempt && Help.Strength == _appliedHelp.Strength) return;
            _appliedHelp = Help;
            var strength = 1f + (StrengthHelpFactor - 1f) * Help.Strength;
            var exemptStiffness = Mathf.Lerp(1f, ExemptStiffnessFactor, Help.Exempt);
            for (var i = 1; i < _links.Length; i++)
            {
                var torque = _realisticTorque[i] * Mathf.Max(_tone[i], 0.02f) * strength;
                var stiffnessSi = torque / SaturationRadians * exemptStiffness;
                // Relaxed muscle and joints keep passive damping: damping doesn't fall below
                // that of PassiveDampingTone, so a limp body comes to rest instead of flopping.
                var dampingSi = Mathf.Max(stiffnessSi, _realisticTorque[i] * PassiveDampingTone * strength / SaturationRadians * exemptStiffness) * DampingSeconds;
                var limit = Help.Exempt > 0 ? Mathf.Lerp(torque, torque * 1000f, Help.Exempt) : torque;
                var link = _links[i];
                link.xDrive = Motor(link.xDrive, stiffnessSi, dampingSi, limit);
                link.yDrive = Motor(link.yDrive, stiffnessSi, dampingSi, limit);
                link.zDrive = Motor(link.zDrive, stiffnessSi, dampingSi, limit);
            }
        }

        private static ArticulationDrive Motor(ArticulationDrive drive, float stiffnessSi, float dampingSi, float limit)
        {
            drive.driveType = ArticulationDriveType.Force;
            drive.stiffness = stiffnessSi * DriveUnitsPerSi;
            drive.damping = dampingSi * DriveUnitsPerSi;
            drive.forceLimit = limit;
            return drive;
        }

        private static ArticulationDrive Drive((float Lower, float Upper) limits) => new()
        {
            lowerLimit = limits.Lower, upperLimit = limits.Upper, driveType = ArticulationDriveType.Force,
        };

        // ---- Body tables ---------------------------------------------------------------------

        /// <summary>Joint limits in degrees about Unity's X, Y and Z axes in the joint's rest
        /// frame, for the left side; the right side mirrors Y and Z. Each main bend reaches at
        /// least M0's placeholder maximum (canonical_skeleton.json) where that is anatomical: hip,
        /// ankle, toes, wrist, spine and neck. The knee stops at 150° (M0: 160°, the passive limit
        /// that squashes soft tissue the collision shapes don't have; at 160° a fallen body wedged
        /// its foot and never settled). M0's values for the head, collarbones and upper arms go
        /// past anatomy and wait for M1's dataset percentiles. Signs: +X bends the knee and
        /// leans the spine forward; −X lifts the thigh forward; +Y bends the left elbow; −Z raises
        /// the left arm.</summary>
        public static ((float, float) X, (float, float) Y, (float, float) Z) Limits(int joint)
        {
            var name = Canonical.Names[joint];
            var side = name.StartsWith("right", StringComparison.Ordinal) ? "right" : name.StartsWith("left", StringComparison.Ordinal) ? "left" : "";
            var part = side.Length == 0 ? name : char.ToLowerInvariant(name[side.Length]) + name.Substring(side.Length + 1);
            ((float, float), (float, float), (float, float)) l = part switch
            {
                "spine" => ((-30, 60), (-30, 30), (-30, 30)),
                "chest" => ((-20, 45), (-25, 25), (-20, 20)),
                "upperChest" => ((-20, 45), (-25, 25), (-20, 20)),
                "neck" => ((-40, 70), (-60, 60), (-35, 35)),
                "head" => ((-35, 40), (-40, 40), (-25, 25)),
                "shoulder" => ((-15, 15), (-25, 25), (-30, 10)),
                "upperArm" => ((-90, 90), (-60, 110), (-100, 90)),
                "lowerArm" => ((-80, 80), (-5, 150), (-5, 5)),
                "hand" => ((-20, 20), (-35, 35), (-90, 90)),
                "upperLeg" => ((-130, 30), (-40, 40), (-45, 25)),
                "lowerLeg" => ((-3, 150), (-5, 5), (-3, 3)),
                "foot" => ((-30, 70), (-15, 15), (-25, 25)),
                "toes" => ((-60, 20), (-3, 3), (-3, 3)),
                _ => ((-30, 30), (-30, 30), (-30, 30)),
            };
            if (side == "right") l = (l.Item1, (-l.Item2.Item2, -l.Item2.Item1), (-l.Item3.Item2, -l.Item3.Item1));
            return l;
        }

        /// <summary>Realistic peak joint torque per kilogram of body mass (N·m/kg).</summary>
        public static float TorquePerKg(int joint)
        {
            var name = Canonical.Names[joint];
            if (name.EndsWith("UpperLeg")) return 2.5f;
            if (name.EndsWith("LowerLeg")) return 2.2f;
            if (name.EndsWith("Foot")) return 1.6f;
            if (name.EndsWith("Toes")) return 0.3f;
            if (name.EndsWith("Shoulder")) return 0.6f;
            if (name.EndsWith("UpperArm")) return 1.0f;
            if (name.EndsWith("LowerArm")) return 0.7f;
            if (name.EndsWith("Hand")) return 0.15f;
            return name switch
            {
                "spine" => 2.2f, "chest" => 1.8f, "upperChest" => 1.5f, "neck" => 0.5f, "head" => 0.3f, _ => 1f,
            };
        }

        /// <summary>Segment mass fractions (de Leva 1996, female), folded onto the avatar's joints.</summary>
        public static float[] SegmentMasses(AvatarMap map)
        {
            var share = new Dictionary<string, float>
            {
                ["hips"] = 12.47f, ["spine"] = 14.65f, ["chest"] = 7.7f, ["upperChest"] = 6.75f,
                ["neck"] = 1.5f, ["head"] = 5.18f, ["leftShoulder"] = 0.5f, ["rightShoulder"] = 0.5f,
                ["leftUpperArm"] = 2.55f, ["rightUpperArm"] = 2.55f, ["leftLowerArm"] = 1.38f, ["rightLowerArm"] = 1.38f,
                ["leftHand"] = 0.56f, ["rightHand"] = 0.56f, ["leftUpperLeg"] = 14.78f, ["rightUpperLeg"] = 14.78f,
                ["leftLowerLeg"] = 4.81f, ["rightLowerLeg"] = 4.81f, ["leftFoot"] = 1.0f, ["rightFoot"] = 1.0f,
                ["leftToes"] = 0.29f, ["rightToes"] = 0.29f,
            };
            var masses = new float[map.Joints.Count];
            for (var i = 0; i < map.Joints.Count; i++)
                foreach (var c in map.Compose[i]) masses[i] += share[Canonical.Names[c]];
            // A dropped leaf (toes) goes to its parent; a joint folded forward is counted in its child.
            foreach (var d in map.Dropped)
            {
                var parent = map.PresentParent(d);
                masses[Array.IndexOf(map.Joints.ToArray(), parent)] += share[Canonical.Names[d]];
            }
            var total = masses.Sum();
            for (var i = 0; i < masses.Length; i++) masses[i] /= total;
            return masses;
        }

        private List<Vector3>?[] _meshPoints = Array.Empty<List<Vector3>?>();
        /// <summary>Rigid accessories (a robot arm, a weapon, a backpack): vertices that follow a
        /// non-humanoid bone, grouped by the accessory's top bone, per body part they hang from.</summary>
        private List<List<Vector3>>?[] _accessoryPoints = Array.Empty<List<List<Vector3>>?>();

        /// <summary>
        /// The avatar's rest-pose mesh, split by body part (simulation space): every vertex goes to
        /// the part its strongest bone hangs from, including rigid accessories (a robot arm, solid
        /// hair). Hair, clothing and accessories on spring joints are left out: they don't push.
        /// </summary>
        private void MeshPointsByLink()
        {
            var n = _fit.Map.Joints.Count;
            var points = _meshPoints;
            var accessories = new Dictionary<Transform, (int Link, List<Vector3> Points)>();
            // Every humanoid bone (fingers, eyes and jaw included) is body, not an accessory.
            var humanoid = new HashSet<Transform>();
            foreach (HumanBodyBones hb in Enum.GetValues(typeof(HumanBodyBones)))
                if (hb != HumanBodyBones.LastBone && _fit.Animator.GetBoneTransform(hb) is { } human) humanoid.Add(human);
            var boneToLink = new Dictionary<Transform, int>();
            for (var i = 0; i < n; i++)
                if (_fit.Bone(_fit.Map.Joints[i]) is { } bone) boneToLink[bone] = i;
            var springs = new HashSet<Transform>();
            if (_fit.Root.GetComponent<Vrm10Instance>() is { } vrm)
                foreach (var spring in vrm.SpringBone.Springs)
                    foreach (var joint in spring.Joints)
                        if (joint != null) springs.Add(joint.transform);
            var linkOf = new Dictionary<Transform, int>();
            int LinkOf(Transform? b)
            {
                if (b == null) return -1;
                if (linkOf.TryGetValue(b, out var cached)) return cached;
                var found = springs.Contains(b) ? -1 : boneToLink.TryGetValue(b, out var direct) ? direct : LinkOf(b.parent);
                linkOf[b] = found;
                return found;
            }
            // The top non-humanoid bone between a vertex's bone and its body part (null: body).
            Transform? AccessoryRoot(Transform b)
            {
                Transform? top = null;
                for (var x = b; x != null && !boneToLink.ContainsKey(x); x = x.parent)
                {
                    if (humanoid.Contains(x)) return null;
                    top = x;
                }
                return top;
            }
            // At rest each vertex sits where its strongest bone's world matrix and bind pose put it:
            // exact for the rest pose, and correct at any display scale (no baking needed).
            var vertices = new List<Vector3>();
            var weights = new List<BoneWeight>();
            var bindposes = new List<Matrix4x4>();
            foreach (var renderer in _fit.Root.GetComponentsInChildren<SkinnedMeshRenderer>())
            {
                var mesh = renderer.sharedMesh;
                if (!renderer.enabled || mesh == null) continue;
                mesh.GetVertices(vertices);
                mesh.GetBoneWeights(weights);
                mesh.GetBindposes(bindposes);
                var bones = renderer.bones;
                var stride = Math.Max(1, vertices.Count / 6000); // plenty for percentiles
                for (var v = 0; v < vertices.Count && v < weights.Count; v += stride)
                {
                    var index = weights[v].boneIndex0;
                    if (index >= bones.Length || index >= bindposes.Count || bones[index] == null) continue;
                    var link = LinkOf(bones[index]);
                    if (link < 0) continue;
                    var world = (bones[index].localToWorldMatrix * bindposes[index]).MultiplyPoint3x4(vertices[v]);
                    var point = SimPoint(_fit.RootSpace(world));
                    if (AccessoryRoot(bones[index]) is { } accessory)
                    {
                        if (!accessories.TryGetValue(accessory, out var group)) accessories[accessory] = group = (link, new List<Vector3>());
                        group.Points.Add(point);
                    }
                    else (points[link] ??= new List<Vector3>()).Add(point);
                }
            }
            foreach (var (link, group) in accessories.Values)
                if (group.Count >= 30) (_accessoryPoints[link] ??= new List<List<Vector3>>()).Add(group);
        }

        private static float Percentile(List<float> values, float q)
        {
            values.Sort();
            return values[Mathf.Clamp(Mathf.RoundToInt(q * (values.Count - 1)), 0, values.Count - 1)];
        }

        /// <summary>Skin and clothing against a desk or taskbar: grippier than Unity's default
        /// (0.6), so a limp hand or foot stops instead of creeping, and it doesn't bounce.</summary>
        public static readonly PhysicsMaterial SkinMaterial = new("skin")
        {
            staticFriction = 1.0f, dynamicFriction = 0.9f, bounciness = 0f,
            frictionCombine = PhysicsMaterialCombine.Average, bounceCombine = PhysicsMaterialCombine.Minimum,
        };

        private IEnumerable<Collider> BuildColliders(GameObject go, int i, int joint, Vector3[] rest, float h)
        {
            foreach (var c in BuildBodyColliders(go, i, joint, rest, h)) yield return c;
            if (_accessoryPoints.Length <= i || _accessoryPoints[i] is not { } groups) yield break;
            // Each rigid accessory gets a box of its own: 2nd to 98th percentile of its vertices,
            // in the part's frame, at most 0.8 of her height on any side.
            foreach (var group in groups)
            {
                var lo = Vector3.zero; var hi = Vector3.zero;
                for (var axis = 0; axis < 3; axis++)
                {
                    var values = group.Select(q => (Quaternion.Inverse(_frame) * q)[axis]).ToList();
                    lo[axis] = Percentile(values, 0.02f);
                    hi[axis] = Percentile(values, 0.98f);
                }
                var size = Vector3.Min(Vector3.Max(hi - lo, Vector3.one * 0.01f * h), Vector3.one * 0.8f * h);
                var box = go.AddComponent<BoxCollider>();
                box.center = go.transform.InverseTransformPoint(_frame * ((lo + hi) / 2));
                box.size = size;
                yield return box;
            }
        }

        private IEnumerable<Collider> BuildBodyColliders(GameObject go, int i, int joint, Vector3[] rest, float h)
        {
            var points = _meshPoints.Length > i && _meshPoints[i] is { Count: >= 30 } list ? list : null;
            var name = Canonical.Names[joint];
            var t = go.transform;
            Vector3 Local(Vector3 world) => t.InverseTransformPoint(world);
            Vector3? Child(params string[] names)
            {
                foreach (var n in names)
                {
                    var k = _fit.IndexOf(Canonical.Index(n));
                    if (k >= 0) return rest[k];
                }
                return null;
            }
            var self = rest[i];
            // Fitted to the mesh: the radius that covers 95% of this part's vertices around the
            // bone, between 0.6 and 4 times the proportional guess, and the far end out to where
            // the vertices reach (hands past their joints).
            Collider Fitted(Vector3 from, Vector3 to, float guess, bool extend = false)
            {
                if (points == null) return Capsule(from, to, guess);
                var axis = to - from;
                var length = Mathf.Max(axis.magnitude, 1e-4f);
                var direction = axis / length;
                var distances = new List<float>(points.Count);
                var along = new List<float>(points.Count);
                foreach (var q in points)
                {
                    var s = Mathf.Clamp(Vector3.Dot(q - from, direction), 0f, extend ? float.MaxValue : length);
                    distances.Add(Vector3.Distance(q, from + direction * s));
                    along.Add(Vector3.Dot(q - from, direction));
                }
                var radius = Mathf.Clamp(Percentile(distances, 0.95f), 0.6f * guess, 4f * guess);
                var end = extend ? Mathf.Max(Percentile(along, 0.98f) - radius, length * 0.3f) : length;
                return Capsule(from, from + direction * end, radius);
            }
            Collider Capsule(Vector3 from, Vector3 to, float radius)
            {
                var c = go.AddComponent<CapsuleCollider>();
                var a = Local(from); var b = Local(to);
                var axis = b - a;
                c.center = (a + b) / 2;
                c.radius = radius;
                var ax = new Vector3(Mathf.Abs(axis.x), Mathf.Abs(axis.y), Mathf.Abs(axis.z));
                c.direction = ax.x >= ax.y && ax.x >= ax.z ? 0 : ax.y >= ax.z ? 1 : 2;
                c.height = axis.magnitude + 2 * radius;
                return c;
            }
            // Fitted to the mesh: the box between the 2nd and 98th percentile of this part's
            // vertices on each axis, between 0.6 and 3 times the proportional guess.
            Collider FittedBox(Vector3 centerWorld, Vector3 guess, float? floor = null)
            {
                if (points == null) return Box(centerWorld, guess);
                var lo = Vector3.zero; var hi = Vector3.zero;
                for (var axis = 0; axis < 3; axis++)
                {
                    var values = points.Select(q => (Quaternion.Inverse(_frame) * q)[axis]).ToList();
                    lo[axis] = Percentile(values, 0.02f);
                    hi[axis] = Percentile(values, 0.98f);
                }
                var guessLocal = Quaternion.Inverse(_frame) * centerWorld;
                var size = hi - lo;
                var center = (lo + hi) / 2;
                for (var axis = 0; axis < 3; axis++)
                {
                    var clamped = Mathf.Clamp(size[axis], 0.6f * guess[axis], 3f * guess[axis]);
                    if (Mathf.Abs(clamped - size[axis]) > 1e-5f) center[axis] = guessLocal[axis];
                    size[axis] = clamped;
                }
                if (floor is float ground)
                {
                    // Feet keep their soles on the ground they stand on at rest.
                    var top = center.y + size.y / 2;
                    var bottom = (Quaternion.Inverse(_frame) * new Vector3(0, ground, 0)).y;
                    center.y = (top + bottom) / 2;
                    size.y = Mathf.Max(top - bottom, 0.02f * h);
                }
                return Box(_frame * center, size);
            }
            Collider Box(Vector3 centerWorld, Vector3 size)
            {
                var c = go.AddComponent<BoxCollider>();
                c.center = Local(centerWorld);
                c.size = size;
                return c;
            }
            var legL = SimPoint(_fit.RestPosition(_fit.IndexOf(Canonical.Index("leftUpperLeg"))));
            var legR = SimPoint(_fit.RestPosition(_fit.IndexOf(Canonical.Index("rightUpperLeg"))));
            var hipSpan = Vector3.Distance(legL, legR);
            switch (name)
            {
                case "hips":
                    yield return FittedBox((self + (legL + legR) / 2) / 2, new Vector3(hipSpan + 0.07f * h, 0.09f * h, 0.105f * h));
                    break;
                case "spine": case "chest": case "upperChest":
                {
                    var top = Child(name == "spine" ? new[] { "chest", "upperChest", "neck", "head" }
                                  : name == "chest" ? new[] { "upperChest", "neck", "head" } : new[] { "neck", "head" })!.Value;
                    var width = name == "spine" ? hipSpan + 0.04f * h : hipSpan + 0.07f * h;
                    var len = Mathf.Max(top.y - self.y, 0.02f * h);
                    yield return FittedBox((self + top) / 2, new Vector3(width, len, 0.1f * h));
                    break;
                }
                case "neck":
                    yield return Fitted(self, Child("head")!.Value, 0.028f * h);
                    break;
                case "head":
                {
                    if (points != null)
                    {
                        // The head and anything rigid on it (solid hair, ears, a helmet).
                        yield return FittedBox(self + _frame * new Vector3(0, 0.06f * h, 0.012f * h), new Vector3(0.13f * h, 0.14f * h, 0.13f * h));
                        break;
                    }
                    var c = go.AddComponent<SphereCollider>();
                    c.radius = 0.068f * h;
                    c.center = Local(self + _frame * new Vector3(0, 0.06f * h, 0.012f * h));
                    yield return c;
                    break;
                }
                default:
                {
                    var side = name.StartsWith("left", StringComparison.Ordinal) ? "left" : "right";
                    var part = name.Substring(side.Length);
                    switch (part)
                    {
                        case "Shoulder":
                            yield return Fitted(self, Child(side + "UpperArm")!.Value, 0.024f * h);
                            break;
                        case "UpperArm":
                            yield return Fitted(self, Child(side + "LowerArm")!.Value, 0.027f * h);
                            break;
                        case "LowerArm":
                            yield return Fitted(self, Child(side + "Hand")!.Value, 0.021f * h);
                            break;
                        case "Hand":
                        {
                            var forearm = self - rest[_fit.IndexOf(Canonical.Index(side + "LowerArm"))];
                            yield return Fitted(self, self + forearm.normalized * 0.07f * h, 0.02f * h, extend: true);
                            break;
                        }
                        // Limb thickness follows her width as well as her height, so a tall,
                        // narrow avatar's thighs don't overlap.
                        case "UpperLeg":
                            yield return Fitted(self, Child(side + "LowerLeg")!.Value, Mathf.Min(0.048f * h, 0.36f * hipSpan));
                            break;
                        case "LowerLeg":
                            yield return Fitted(self, Child(side + "Foot")!.Value, Mathf.Min(0.034f * h, 0.26f * hipSpan));
                            break;
                        case "Foot":
                        {
                            // From the heel to the toes (or the toe tip if the avatar has no toes),
                            // sitting on the ground in the rest pose.
                            var forward = _frame * Vector3.forward;
                            var toes = Child(side + "Toes");
                            var tip = toes ?? self + forward * 0.11f * h;
                            var heel = self - forward * 0.035f * h;
                            var ground = _ground;
                            var height = Mathf.Max(self.y - ground, 0.02f * h);
                            var center = new Vector3((heel.x + tip.x) / 2, ground + height / 2, (heel.z + tip.z) / 2);
                            yield return FittedBox(center, new Vector3(0.055f * h, height, Vector3.Distance(Flat(heel), Flat(tip))), ground);
                            break;
                        }
                        case "Toes":
                        {
                            var forward = _frame * Vector3.forward;
                            var ground = _ground;
                            var height = Mathf.Max(self.y - ground, 0.012f * h);
                            var center = new Vector3(self.x, ground + height / 2, self.z) + forward * 0.025f * h;
                            yield return FittedBox(center, new Vector3(0.055f * h, height, 0.05f * h), ground);
                            break;
                        }
                    }
                    break;
                }
            }
        }

        private static Vector3 Flat(Vector3 v) => new(v.x, 0, v.z);

        /// <summary>
        /// Self-collision only between the two thighs (so the legs can't pass through each other).
        /// Shins and feet don't: fitted to the mesh (boots, shoes) they press together when she lies
        /// down, and light, relaxed limbs wedged against each other buzz instead of settling. Arms
        /// against the torso and thighs against the trunk would be shoved apart every step in
        /// ordinary poses (arms hanging, sitting), as in most ragdolls.
        /// </summary>
        private static void IgnoreSelfOverlaps(List<Collider> colliders)
        {
            Physics.SyncTransforms();
            static int Leg(Collider c)
            {
                var n = c.attachedArticulationBody != null ? c.attachedArticulationBody.name : c.name;
                if (!n.EndsWith("UpperLeg")) return 0;
                return n.Contains("left") ? 1 : 2;
            }
            for (var a = 0; a < colliders.Count; a++)
            for (var b = a + 1; b < colliders.Count; b++)
            {
                var la = Leg(colliders[a]); var lb = Leg(colliders[b]);
                var oppositeLegs = la != 0 && lb != 0 && la != lb;
                // Legs already touching at rest would be pushed apart every step.
                var touching = oppositeLegs && Physics.ComputePenetration(colliders[a], colliders[a].transform.position, colliders[a].transform.rotation,
                    colliders[b], colliders[b].transform.position, colliders[b].transform.rotation, out _, out _);
                if (!oppositeLegs || touching) Physics.IgnoreCollision(colliders[a], colliders[b], true);
            }
        }

        public void Dispose()
        {
            if (Container != null) UnityEngine.Object.Destroy(Container);
        }
    }
}
