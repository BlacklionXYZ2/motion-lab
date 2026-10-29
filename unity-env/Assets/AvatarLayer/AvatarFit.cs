using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace MotionEnv.AvatarLayer
{
    /// <summary>
    /// A2, fitting: puts a canonical pose onto one particular VRM avatar.
    /// <list type="bullet">
    /// <item>Rotations: folded by <see cref="AvatarMap"/>, then retargeted from world-aligned rest
    /// frames onto the avatar's own bone frames (the same formula as UniVRM's control rig).</item>
    /// <item>Proportions: hips travel and height scale by leg length, so feet don't slide on an
    /// avatar with longer or shorter legs than the canonical (Alice's) ones.</item>
    /// <item>Contacts: a planted foot is pinned where it landed with two-bone leg IK.</item>
    /// </list>
    /// Construct it while the avatar is in its rest (T-) pose.
    /// </summary>
    public sealed class AvatarFit
    {
        private readonly Transform _root;
        private readonly Transform[] _bones;             // per avatar joint
        private readonly Quaternion[] _initLocal, _initRootSpace;
        private readonly Vector3 _hipsRest;              // root space
        private readonly Vector3[] _restPositions;       // root space, per avatar joint
        private readonly Quaternion[] _avatarLocal;
        private readonly Vector3?[] _pins = new Vector3?[2];

        public AvatarMap Map { get; }
        public Animator Animator { get; }
        public Transform Root => _root;
        /// <summary>The avatar's leg length (upper leg to foot) in its own units.</summary>
        public float LegLength { get; }
        /// <summary>Avatar leg length over the canonical leg length.</summary>
        public float RootScale { get; }
        public float HipsRestHeight => _hipsRest.y;
        public float Height { get; }
        /// <summary>The lowest foot or toe joint at rest (root space): about where the sole is.</summary>
        public float SoleHeight { get; }
        /// <summary>Horizontal drift of each planted foot since it was planted (left, right).</summary>
        public float[] PinnedDrift { get; } = new float[2];
        /// <summary>
        /// Ways the avatar's rest pose isn't the T-pose the canonical skeleton assumes (VRM
        /// requires one). Empty for a proper T-pose. Every rotation is relative to rest, so a
        /// model that rests in, say, an A-pose would hold every arm pose that much lower.
        /// </summary>
        public IReadOnlyList<string> RestPoseIssues { get; }

        public AvatarFit(Animator animator)
        {
            Animator = animator != null ? animator : throw new ArgumentNullException(nameof(animator));
            _root = animator.transform;
            var present = new List<int>();
            for (var j = 0; j < Canonical.Count; j++)
                if (animator.GetBoneTransform(Canonical.Bones[j]) != null) present.Add(j);
            Map = AvatarMap.ForJoints(present);
            var n = Map.Joints.Count;
            _bones = new Transform[n];
            _initLocal = new Quaternion[n];
            _initRootSpace = new Quaternion[n];
            _avatarLocal = new Quaternion[n];
            _restPositions = new Vector3[n];
            for (var i = 0; i < n; i++)
            {
                var bone = animator.GetBoneTransform(Canonical.Bones[Map.Joints[i]]);
                _bones[i] = bone;
                _initLocal[i] = bone.localRotation;
                _initRootSpace[i] = Quaternion.Inverse(_root.rotation) * bone.rotation;
                _restPositions[i] = RootSpace(bone.position);
            }
            _hipsRest = RootSpace(Bone(0).position);
            float Leg(HumanBodyBones upper, HumanBodyBones lower, HumanBodyBones foot)
            {
                Vector3 P(HumanBodyBones b) => RootSpace(animator.GetBoneTransform(b).position);
                return Vector3.Distance(P(upper), P(lower)) + Vector3.Distance(P(lower), P(foot));
            }
            LegLength = 0.5f * (Leg(HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftFoot) +
                                Leg(HumanBodyBones.RightUpperLeg, HumanBodyBones.RightLowerLeg, HumanBodyBones.RightFoot));
            RootScale = LegLength / Canonical.LegLength;
            var head = RootSpace(animator.GetBoneTransform(HumanBodyBones.Head).position);
            var sole = float.MaxValue;
            foreach (var b in new[] { HumanBodyBones.LeftFoot, HumanBodyBones.RightFoot, HumanBodyBones.LeftToes, HumanBodyBones.RightToes })
                if (animator.GetBoneTransform(b) is { } t) sole = Mathf.Min(sole, RootSpace(t.position).y);
            // Toe joints sit on the sole; the top of the head is about 9% above the head joint.
            Height = (head.y - sole) * 1.09f;
            SoleHeight = sole;
            RestPoseIssues = CheckRestPose(animator);
        }

        /// <summary>Limbs and spine must point along the canonical rest directions (within 20°
        /// for arms, 15° for legs and spine): arms out sideways, legs and spine vertical, facing
        /// +Z with her left at −X.</summary>
        private List<string> CheckRestPose(Animator animator)
        {
            var issues = new List<string>();
            void Check(HumanBodyBones from, HumanBodyBones to, Vector3 expected, float tolerance, string what)
            {
                var a = animator.GetBoneTransform(from);
                var b = animator.GetBoneTransform(to);
                if (a == null || b == null) return;
                var direction = RootSpace(b.position) - RootSpace(a.position);
                var angle = Vector3.Angle(direction, expected);
                if (angle > tolerance) issues.Add($"{what} is {angle:0}° from the T-pose direction");
            }
            Check(HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm, Vector3.left, 20f, "the left upper arm");
            Check(HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm, Vector3.right, 20f, "the right upper arm");
            Check(HumanBodyBones.LeftLowerArm, HumanBodyBones.LeftHand, Vector3.left, 20f, "the left forearm");
            Check(HumanBodyBones.RightLowerArm, HumanBodyBones.RightHand, Vector3.right, 20f, "the right forearm");
            Check(HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg, Vector3.down, 15f, "the left thigh");
            Check(HumanBodyBones.RightUpperLeg, HumanBodyBones.RightLowerLeg, Vector3.down, 15f, "the right thigh");
            Check(HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftFoot, Vector3.down, 15f, "the left shin");
            Check(HumanBodyBones.RightLowerLeg, HumanBodyBones.RightFoot, Vector3.down, 15f, "the right shin");
            Check(HumanBodyBones.Hips, HumanBodyBones.Head, Vector3.up, 15f, "the spine");
            return issues;
        }

        /// <summary>The avatar's transform for a canonical joint, or null if it lacks it.</summary>
        public Transform? Bone(int canonical)
        {
            var i = IndexOf(canonical);
            return i < 0 ? null : _bones[i];
        }

        public int IndexOf(int canonical)
        {
            for (var i = 0; i < Map.Joints.Count; i++) if (Map.Joints[i] == canonical) return i;
            return -1;
        }

        public Vector3 RootSpace(Vector3 world) => Quaternion.Inverse(_root.rotation) * (world - _root.position) /
                                                    Mathf.Max(_root.lossyScale.y, 1e-6f);
        public Vector3 WorldSpace(Vector3 rootSpace) => _root.position + _root.rotation * (rootSpace * _root.lossyScale.y);

        /// <summary>The avatar's rest position of an avatar joint (by index), in root space.</summary>
        public Vector3 RestPosition(int avatarJoint) => _restPositions[avatarJoint];

        /// <summary>Where the avatar's hips go for a canonical root position.</summary>
        public Vector3 HipsFor(Vector3 canonicalRoot)
        {
            var fromRest = canonicalRoot - new Vector3(0, Canonical.HipsHeight, 0);
            return _hipsRest + fromRest * RootScale;
        }

        /// <summary>Folds a canonical pose to the avatar's joints (normalized local rotations).</summary>
        public Quaternion[] Fold(CanonicalPose pose) => Fold(pose, new Quaternion[Map.Joints.Count]);

        /// <summary>Folds into an existing array (no allocation; for per-step use).</summary>
        public Quaternion[] Fold(CanonicalPose pose, Quaternion[] output)
        {
            Map.ToAvatar(pose.Local, output);
            return output;
        }

        /// <summary>Applies a canonical pose: rotations, scaled root, then foot pins.</summary>
        public void Apply(CanonicalPose pose, bool pinFeet = true)
        {
            Map.ToAvatar(pose.Local, _avatarLocal);
            ApplyNormalized(_avatarLocal, HipsFor(pose.Root));
            if (pinFeet) PinFeet(pose.Contacts[0] || pose.Contacts[1], pose.Contacts[2] || pose.Contacts[3]);
        }

        /// <summary>Normalized local rotations (per avatar joint) and the hips position (root space).</summary>
        public void ApplyNormalized(Quaternion[] normalizedLocal, Vector3 hipsRootSpace)
        {
            _bones[0].position = WorldSpace(hipsRootSpace);
            for (var i = 0; i < _bones.Length; i++)
            {
                var init = _initRootSpace[i];
                _bones[i].localRotation = _initLocal[i] * (Quaternion.Inverse(init) * normalizedLocal[i] * init);
            }
        }

        /// <summary>Reads the avatar's current pose back as normalized local rotations.</summary>
        public void ReadNormalized(Quaternion[] normalizedLocal)
        {
            for (var i = 0; i < _bones.Length; i++)
            {
                var init = _initRootSpace[i];
                normalizedLocal[i] = init * (Quaternion.Inverse(_initLocal[i]) * _bones[i].localRotation) * Quaternion.Inverse(init);
            }
        }

        private void PinFeet(bool leftPlanted, bool rightPlanted)
        {
            Pin(0, leftPlanted, HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftFoot);
            Pin(1, rightPlanted, HumanBodyBones.RightUpperLeg, HumanBodyBones.RightLowerLeg, HumanBodyBones.RightFoot);
        }

        private void Pin(int side, bool planted, HumanBodyBones upper, HumanBodyBones lower, HumanBodyBones foot)
        {
            var footBone = Animator.GetBoneTransform(foot);
            if (!planted) { _pins[side] = null; PinnedDrift[side] = 0; return; }
            if (_pins[side] is not Vector3 pin) { _pins[side] = footBone.position; return; }
            var before = footBone.position;
            var footRotation = footBone.rotation;
            TwoBoneIk.Solve(Animator.GetBoneTransform(upper), Animator.GetBoneTransform(lower), footBone, pin,
                _root.forward);
            footBone.rotation = footRotation;
            var drift = footBone.position - pin;
            drift.y = 0;
            PinnedDrift[side] = drift.magnitude / Mathf.Max(_root.lossyScale.y, 1e-6f);
        }
    }

    /// <summary>Two-bone IK (as in Unity's Animation Rigging): bends the middle joint, then aims.</summary>
    public static class TwoBoneIk
    {
        /// <param name="bendHint">The direction the middle joint should point (a knee: forward).</param>
        public static void Solve(Transform a, Transform b, Transform c, Vector3 target, Vector3 bendHint, float weight = 1f)
        {
            var aPos = a.position;
            var ab = b.position - aPos;
            var bc = c.position - b.position;
            var ac = c.position - aPos;
            var at = target - aPos;
            var abLen = ab.magnitude;
            var bcLen = bc.magnitude;
            var acLen = ac.magnitude;
            var atLen = Mathf.Clamp(at.magnitude, 1e-4f, abLen + bcLen - 1e-4f);
            var oldAngle = Triangle(acLen, abLen, bcLen);
            var newAngle = Triangle(atLen, abLen, bcLen);
            var axis = Vector3.Cross(ab, bc);
            // A straight limb has no bend plane: bend so the middle joint moves towards the hint.
            if (axis.sqrMagnitude < 1e-8f) axis = Vector3.Cross(bendHint, ab);
            if (axis.sqrMagnitude < 1e-8f) axis = Vector3.Cross(Vector3.forward, ab);
            axis.Normalize();
            var half = 0.5f * (oldAngle - newAngle) * weight;
            var delta = new Quaternion(axis.x * Mathf.Sin(half), axis.y * Mathf.Sin(half), axis.z * Mathf.Sin(half), Mathf.Cos(half));
            b.rotation = delta * b.rotation;
            ac = c.position - aPos;
            a.rotation = Quaternion.Slerp(Quaternion.identity, Quaternion.FromToRotation(ac, at), weight) * a.rotation;
        }

        private static float Triangle(float opposite, float side1, float side2)
        {
            var cos = (side1 * side1 + side2 * side2 - opposite * opposite) / Mathf.Max(2f * side1 * side2, 1e-8f);
            return Mathf.Acos(Mathf.Clamp(cos, -1f, 1f));
        }
    }
}
