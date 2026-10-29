using System;
using UnityEngine;

namespace MotionEnv.AvatarLayer
{
    /// <summary>
    /// The motion core's 22-joint canonical skeleton (M0, <c>versions/vN/motionlab/skeleton.py</c>), in
    /// Unity terms. A test checks these tables against <c>canonical_skeleton.json</c>.
    /// <para>Frames: motion-lab is right-handed (+Y up, she faces +Z, her left is +X); Unity is
    /// left-handed (+Y up, she faces +Z, her left is −X). Converting mirrors X.</para>
    /// <para>Every joint's rest frame is world-aligned, so the rest pose is identity everywhere and
    /// a "normalized local rotation" is the same thing in both the model and the avatar layer.</para>
    /// </summary>
    public static class Canonical
    {
        public const int Count = 22;

        public static readonly string[] Names =
        {
            "hips", "spine", "chest", "upperChest", "neck", "head",
            "leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand",
            "rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand",
            "leftUpperLeg", "leftLowerLeg", "leftFoot", "leftToes",
            "rightUpperLeg", "rightLowerLeg", "rightFoot", "rightToes",
        };

        public static readonly int[] Parents =
            { -1, 0, 1, 2, 3, 4, 3, 6, 7, 8, 3, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20 };

        public static readonly HumanBodyBones[] Bones =
        {
            HumanBodyBones.Hips, HumanBodyBones.Spine, HumanBodyBones.Chest, HumanBodyBones.UpperChest,
            HumanBodyBones.Neck, HumanBodyBones.Head,
            HumanBodyBones.LeftShoulder, HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm, HumanBodyBones.LeftHand,
            HumanBodyBones.RightShoulder, HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm, HumanBodyBones.RightHand,
            HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftFoot, HumanBodyBones.LeftToes,
            HumanBodyBones.RightUpperLeg, HumanBodyBones.RightLowerLeg, HumanBodyBones.RightFoot, HumanBodyBones.RightToes,
        };

        /// <summary>Largest bend from rest, degrees (M0 placeholders until M1 measures datasets).</summary>
        public static readonly float[] MaxAngleDegrees =
            { 0, 60, 45, 45, 70, 80, 40, 170, 150, 90, 40, 170, 150, 90, 130, 160, 70, 60, 130, 160, 70, 60 };

        /// <summary>The VRM joints an avatar may lack (VRM 1.0 humanoid).</summary>
        public static readonly string[] Optional =
            { "chest", "upperChest", "neck", "leftShoulder", "rightShoulder", "leftToes", "rightToes" };

        public static readonly int[] Mirror = { 0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 6, 7, 8, 9, 18, 19, 20, 21, 14, 15, 16, 17 };

        /// <summary>Alice's leg (upper leg to foot), the canonical proportions, in metres.</summary>
        public const float LegLength = 0.3710f + 0.3855f;
        /// <summary>Alice's hips height in the canonical rest pose, in metres.</summary>
        public const float HipsHeight = 0.894626f;

        public static int Index(string name)
        {
            var i = Array.IndexOf(Names, name);
            if (i < 0) throw new ArgumentException("not a canonical joint: " + name);
            return i;
        }

        public static bool IsLeft(int joint) => Names[joint].StartsWith("left", StringComparison.Ordinal);

        // motion-lab (right-handed, left = +X) ↔ Unity (left-handed, left = −X): mirror X.
        public static Vector3 ToUnity(Vector3 v) => new(-v.x, v.y, v.z);
        public static Vector3 FromUnity(Vector3 v) => new(-v.x, v.y, v.z);
        public static Quaternion ToUnity(Quaternion q) => new(q.x, -q.y, -q.z, q.w);
        public static Quaternion FromUnity(Quaternion q) => new(q.x, -q.y, -q.z, q.w);
    }

    /// <summary>One tick of motion in the canonical skeleton, in Unity's frame.</summary>
    public sealed class CanonicalPose
    {
        /// <summary>Local rotations relative to rest, in world-aligned rest frames.</summary>
        public readonly Quaternion[] Local = new Quaternion[Canonical.Count];
        /// <summary>Hips position in canonical metres (Alice's proportions).</summary>
        public Vector3 Root = new(0, Canonical.HipsHeight, 0);
        /// <summary>Foot contacts: left foot, left toes, right foot, right toes.</summary>
        public readonly bool[] Contacts = new bool[4];

        public CanonicalPose()
        {
            for (var j = 0; j < Canonical.Count; j++) Local[j] = Quaternion.identity;
        }

        public CanonicalPose CopyFrom(CanonicalPose other)
        {
            Array.Copy(other.Local, Local, Canonical.Count);
            Array.Copy(other.Contacts, Contacts, 4);
            Root = other.Root;
            return this;
        }

        public void Set(string joint, Vector3 eulerDegrees) => Local[Canonical.Index(joint)] = Quaternion.Euler(eulerDegrees);
    }
}
