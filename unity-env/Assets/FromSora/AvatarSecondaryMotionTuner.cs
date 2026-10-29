using System;
using UniVRM10;
using UnityEngine;

namespace Sora.Embodiment.Avatar
{
    public readonly struct StableSpringBoneSettings
    {
        public float Stiffness { get; }
        public float GravityPower { get; }
        public Vector3 GravityDirection { get; }
        public float Drag { get; }
        public float Radius { get; }

        public StableSpringBoneSettings(float stiffness, float gravityPower,
            Vector3 gravityDirection, float drag, float radius)
        {
            Stiffness = stiffness;
            GravityPower = gravityPower;
            GravityDirection = gravityDirection;
            Drag = drag;
            Radius = radius;
        }
    }

    /// <summary>Applies finite runtime-only bounds to imported VRM spring joints.</summary>
    public sealed class AvatarSecondaryMotionTuner : MonoBehaviour
    {
        private Vrm10Instance? _instance;
        public int TunedJointCount { get; private set; }
        /// <summary>Each spring joint's classification (hair, clothing or accessory).</summary>
        public System.Collections.Generic.IReadOnlyDictionary<VRM10SpringBoneJoint, SecondaryClassification> Parts { get; private set; } =
            new System.Collections.Generic.Dictionary<VRM10SpringBoneJoint, SecondaryClassification>();

        /// <summary>
        /// Bounds every spring joint, and (motion roadmap MA, A1) classifies each chain as hair,
        /// clothing or an accessory with its own physics, adding body colliders where the avatar
        /// has none. <paramref name="partsOverridePath"/> is the optional <c>.parts.txt</c> file
        /// beside the avatar.
        /// </summary>
        public int Configure(Vrm10Instance instance, string? partsOverridePath = null, bool generateColliders = true)
        {
            _instance = instance ?? throw new ArgumentNullException(nameof(instance));
            TunedJointCount = 0;
            var animator = instance.GetComponent<Animator>();
            var parts = animator != null && animator.isHuman
                ? AvatarSecondaryParts.Prepare(instance, animator, AvatarSecondaryParts.ReadOverrides(partsOverridePath), generateColliders)
                : new System.Collections.Generic.Dictionary<VRM10SpringBoneJoint, SecondaryClassification>();
            Parts = parts;
            foreach (var joint in instance.GetComponentsInChildren<VRM10SpringBoneJoint>(true))
            {
                var settings = Stabilize(joint.m_stiffnessForce, joint.m_gravityPower,
                    joint.m_gravityDir, joint.m_dragForce, joint.m_jointRadius);
                if (parts.TryGetValue(joint, out var part)) settings = AvatarSecondaryParts.Shape(part, settings);
                // Written onto the joint itself, then the simulation is rebuilt from it: settings
                // pushed into the running buffers are lost when they are rebuilt (2026-10-15).
                joint.m_stiffnessForce = settings.Stiffness;
                joint.m_gravityPower = settings.GravityPower;
                joint.m_gravityDir = settings.GravityDirection;
                joint.m_dragForce = settings.Drag;
                joint.m_jointRadius = settings.Radius;
                joint.m_pitch = Mathf.Clamp(joint.m_pitch, 0, Mathf.PI);
                joint.m_yaw = Mathf.Clamp(joint.m_yaw, 0, Mathf.PI / 2);
                if (!IsFinite(joint.m_limitSpaceOffset)) joint.m_limitSpaceOffset = Quaternion.identity;
                TunedJointCount++;
            }
            if (!instance.Runtime.SpringBone.ReconstructSpringBone())
                Debug.LogWarning("Secondary motion: the spring simulation could not be rebuilt");
            Debug.Log("Secondary motion tuned: " + TunedJointCount + " spring joints");
            return TunedJointCount;
        }

        public void ResetSimulation()
        {
            if (_instance != null) _instance.Runtime.SpringBone.RestoreInitialTransform();
        }

        public static StableSpringBoneSettings Stabilize(float stiffness, float gravityPower,
            Vector3 gravityDirection, float drag, float radius)
        {
            stiffness = IsFinite(stiffness) ? Mathf.Clamp(stiffness, 0, 4f) : 1f;
            gravityPower = IsFinite(gravityPower) ? Mathf.Clamp(gravityPower, 0, 1.5f) : 0;
            drag = IsFinite(drag) ? Mathf.Clamp(drag, 0.35f, 0.9f) : 0.5f;
            radius = IsFinite(radius) ? Mathf.Clamp(radius, 0, 0.2f) : 0.02f;
            if (!IsFinite(gravityDirection) || gravityDirection.sqrMagnitude < 0.000001f)
                gravityDirection = Vector3.down;
            else
                gravityDirection.Normalize();
            return new StableSpringBoneSettings(stiffness, gravityPower, gravityDirection,
                drag, radius);
        }

        private static bool IsFinite(Vector3 value) => IsFinite(value.x) &&
            IsFinite(value.y) && IsFinite(value.z);
        private static bool IsFinite(Quaternion value) => IsFinite(value.x) &&
            IsFinite(value.y) && IsFinite(value.z) && IsFinite(value.w) &&
            value.x * value.x + value.y * value.y + value.z * value.z +
                value.w * value.w > 0.000001f;
        private static bool IsFinite(float value) => !float.IsNaN(value) &&
            !float.IsInfinity(value);
    }
}
