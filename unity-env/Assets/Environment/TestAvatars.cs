using System;
using System.Collections;
using System.IO;
using MotionEnv.AvatarLayer;
using UniVRM10;
using UnityEngine;

namespace MotionEnv.Environment
{
    /// <summary>
    /// The avatars every MA gate runs on: Alice (SORA's default, VRM 0.x, licence forbids copying
    /// her, so she is read from SORA's <c>.local/avatars</c>), Seed-san (VRM 1.0, no upper chest,
    /// bundled with SORA), and "Odd": Seed-san with 30% longer legs, 25% shorter arms and a 15%
    /// longer back, for the swapping gate.
    /// </summary>
    public static class TestAvatars
    {
        public static readonly string[] All = { "Alice", "Seed-san", "Odd" };

        public static string LabRoot => Path.GetFullPath(Path.Combine(Application.dataPath, "../.."));
        public static string SoraRoot => Path.GetFullPath(Path.Combine(LabRoot, "../SORA"));

        /// <summary>The newest model version's folder (<c>versions/vN</c>, highest N): its
        /// <c>motionlab</c> package holds the canonical skeleton and its tests the adapter cases.
        /// Each version is self-contained (owner, 2026-10-25).</summary>
        public static string LatestVersion
        {
            get
            {
                var best = -1;
                string? folder = null;
                foreach (var dir in Directory.GetDirectories(Path.Combine(LabRoot, "versions"), "v*"))
                {
                    if (int.TryParse(Path.GetFileName(dir).Substring(1), out var n) && n > best
                        && Directory.Exists(Path.Combine(dir, "motionlab")))
                    {
                        best = n;
                        folder = dir;
                    }
                }
                return folder ?? throw new DirectoryNotFoundException("no versions/vN/motionlab under " + LabRoot);
            }
        }
        public static string OutputFolder(string name)
        {
            var folder = Path.Combine(LabRoot, "outputs", name);
            Directory.CreateDirectory(folder);
            return folder;
        }

        public static string? PathFor(string avatar) => avatar switch
        {
            "Alice" => Path.Combine(SoraRoot, ".local/avatars/Alice.vrm"),
            "Seed-san" or "Odd" => Path.Combine(SoraRoot, "clients/unity/Assets/StreamingAssets/Seed-san.vrm"),
            _ => null,
        };

        public static bool Available(string avatar) => PathFor(avatar) is { } p && File.Exists(p);

        /// <summary>Loads an avatar at the origin, facing +Z, feet on y = 0, scale 1.</summary>
        public static IEnumerator Load(string avatar, Action<Vrm10Instance> done)
        {
            var path = PathFor(avatar) ?? throw new ArgumentException("unknown test avatar " + avatar);
            var task = Vrm10.LoadPathAsync(path, canLoadVrm0X: true,
                springboneRuntime: new Vrm10FastSpringboneRuntimeStandalone(),
                controlRigGenerationOption: ControlRigGenerationOption.None);
            while (!task.IsCompleted) yield return null;
            if (task.IsFaulted) throw task.Exception!;
            var instance = task.Result;
            instance.name = avatar;
            // The environment drives the VRM itself, in step with physics.
            instance.UpdateType = Vrm10Instance.UpdateTypes.None;
            var animator = instance.GetComponent<Animator>();
            if (avatar == "Odd")
            {
                // Longer legs push the feet below the floor: lift her so they rest where they did.
                var before = LowestFoot(animator);
                MakeOdd(animator);
                instance.transform.position += Vector3.up * (before - LowestFoot(animator));
            }
            done(instance);
        }

        private static float LowestFoot(Animator animator)
        {
            var lowest = float.MaxValue;
            foreach (var b in new[] { HumanBodyBones.LeftFoot, HumanBodyBones.RightFoot, HumanBodyBones.LeftToes, HumanBodyBones.RightToes })
            {
                var t = animator.GetBoneTransform(b);
                if (t != null) lowest = Mathf.Min(lowest, t.position.y);
            }
            return lowest;
        }

        private static void MakeOdd(Animator animator)
        {
            void Scale(HumanBodyBones bone, float factor)
            {
                var t = animator.GetBoneTransform(bone);
                if (t != null) t.localPosition *= factor;
            }
            foreach (var b in new[] { HumanBodyBones.LeftLowerLeg, HumanBodyBones.RightLowerLeg,
                                      HumanBodyBones.LeftFoot, HumanBodyBones.RightFoot }) Scale(b, 1.3f);
            foreach (var b in new[] { HumanBodyBones.LeftLowerArm, HumanBodyBones.RightLowerArm,
                                      HumanBodyBones.LeftHand, HumanBodyBones.RightHand }) Scale(b, 0.75f);
            foreach (var b in new[] { HumanBodyBones.Chest, HumanBodyBones.Neck }) Scale(b, 1.15f);
        }

        /// <summary>A floor (the taskbar's top) at y = 0, as a desktop surface.</summary>
        public static DesktopSurfaces Floor() => DesktopSurfaces.Create(
            new DesktopSurfaces.Surface("taskbar", new Vector3(0, -0.05f, 0), new Vector3(20, 0.1f, 20)));
    }
}
