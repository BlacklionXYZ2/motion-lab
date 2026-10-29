using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using UniVRM10;
using UnityEngine;

namespace Sora.Embodiment.Avatar
{
    /// <summary>What a spring-bone chain is. The body itself is the VRM's humanoid bones.</summary>
    public enum SecondaryPart { Hair, Clothing, Accessory }

    public readonly struct SecondaryClassification
    {
        public SecondaryPart Part { get; }
        /// <summary>A finer kind used for tuning: hair, ponytail, skirt, coat, tie, sleeve, ear, tail, prop, ornament, soft.</summary>
        public string Kind { get; }
        public string Reason { get; }

        public SecondaryClassification(SecondaryPart part, string kind, string reason)
        {
            Part = part; Kind = kind; Reason = reason;
        }
    }

    /// <summary>
    /// Motion roadmap MA, part A1 (owner, 2026-10-15): hair, clothing and accessories move
    /// differently from the body and from each other. VRM separates the body (humanoid bones) from
    /// spring bones but says nothing about what a spring chain is, and converted models (Alice)
    /// mix kinds in one group, with no colliders and zero gravity. So each chain is classified from
    /// its names (VRoid, MMD and romanised Japanese conventions) and where it hangs from, an
    /// optional override file beside the avatar corrects it, and each kind gets its own physics.
    /// Coupling is one way: the body moves these chains, never the reverse.
    /// </summary>
    public static class AvatarSecondaryParts
    {
        private static readonly string[] HairWords =
            { "hair", "kami", "bang", "ahoge", "maegami", "ushirogami", "yokogami", "ponytail", "twintail", "braid", "osage_hair" };
        private static readonly string[] ClothingWords =
            { "skirt", "suka", "coat", "cloak", "cape", "mant", "hood", "sleeve", "sode", "tie", "necktie", "frill",
              "dress", "cloth", "fuku", "jacket", "collar", "apron", "hem", "susomono", "scarf" };
        private static readonly string[] SoftWords = { "bust", "breast", "mune", "oppai" };
        private static readonly string[] EarWords = { "mimi", "ear" };
        private static readonly string[] TailWords = { "tail", "shippo" };
        private static readonly string[] PropWords =
            { "weapon", "wpn", "sword", "gun", "holster", "sheath", "bag", "pouch", "box", "case" };
        private static readonly string[] OrnamentWords =
            { "ornament", "acc", "ribbon", "chain", "earring", "bell", "hat", "pendant", "charm", "strap" };

        /// <summary>Classify one chain from its joint names and the humanoid bone it hangs from.</summary>
        public static SecondaryClassification Classify(IReadOnlyList<string> jointNames, HumanBodyBones? anchor)
        {
            var text = string.Join(" ", jointNames);
            var tokens = Tokens(text);
            bool Has(string[] words) => tokens.Any(t => words.Any(w => w.Length <= 3 ? t == w : t.Contains(w)));
            var onHead = anchor is HumanBodyBones.Head or HumanBodyBones.Neck or HumanBodyBones.Jaw
                or HumanBodyBones.LeftEye or HumanBodyBones.RightEye;
            var onLegs = anchor is HumanBodyBones.Hips or HumanBodyBones.LeftUpperLeg or HumanBodyBones.RightUpperLeg
                or HumanBodyBones.LeftLowerLeg or HumanBodyBones.RightLowerLeg or HumanBodyBones.Spine;
            var where = anchor?.ToString() ?? "no humanoid ancestor";

            if (text.Contains("髪") || Has(HairWords)) return new(SecondaryPart.Hair, "hair", "name");
            if (text.Contains("スカート")) return new(SecondaryPart.Clothing, "skirt", "name");
            if (Has(SoftWords)) return new(SecondaryPart.Accessory, "soft", "name");
            if (Has(ClothingWords))
            {
                var kind = Has(new[] { "skirt", "suka", "hem", "susomono" }) ? "skirt"
                    : Has(new[] { "tie", "necktie", "scarf" }) ? "tie"
                    : Has(new[] { "sleeve", "sode" }) ? "sleeve" : "coat";
                return new(SecondaryPart.Clothing, kind, "name");
            }
            if (Has(EarWords)) return new(SecondaryPart.Accessory, "ear", "name");
            // "Tail" on the head is a ponytail; anywhere else it is a tail.
            if (Has(TailWords))
                return onHead ? new(SecondaryPart.Hair, "ponytail", "name 'tail' on " + where)
                    : new(SecondaryPart.Accessory, "tail", "name 'tail' on " + where);
            if (Has(PropWords)) return new(SecondaryPart.Accessory, "prop", "name");
            if (Has(OrnamentWords)) return new(SecondaryPart.Accessory, "ornament", "name");
            if (onHead) return new(SecondaryPart.Hair, "hair", "hangs from " + where);
            if (onLegs) return new(SecondaryPart.Clothing, "skirt", "hangs from " + where);
            if (anchor is HumanBodyBones.Chest or HumanBodyBones.UpperChest)
                return new(SecondaryPart.Clothing, "coat", "hangs from " + where);
            if (anchor is HumanBodyBones.LeftUpperArm or HumanBodyBones.RightUpperArm
                or HumanBodyBones.LeftLowerArm or HumanBodyBones.RightLowerArm)
                return new(SecondaryPart.Clothing, "sleeve", "hangs from " + where);
            return new(SecondaryPart.Accessory, "ornament", "hangs from " + where);
        }

        /// <summary>Lower-case word pieces: split on non-letters and camelCase boundaries.</summary>
        public static IReadOnlyList<string> Tokens(string text)
        {
            var split = Regex.Replace(text, "([a-z])([A-Z])", "$1 $2");
            return Regex.Split(split.ToLowerInvariant(), "[^a-z]+").Where(t => t.Length > 0).ToArray();
        }

        /// <summary>
        /// The override file beside the avatar, <c>Alice.vrm.parts.txt</c>: one line per chain,
        /// <c>first-joint-name = hair | clothing | accessory</c>; <c>#</c> starts a comment.
        /// </summary>
        public static Dictionary<string, SecondaryPart> ReadOverrides(string? path)
        {
            var overrides = new Dictionary<string, SecondaryPart>(StringComparer.Ordinal);
            if (string.IsNullOrWhiteSpace(path) || !File.Exists(path)) return overrides;
            foreach (var raw in File.ReadAllLines(path))
            {
                var line = raw.Split('#')[0].Trim();
                var at = line.LastIndexOf('=');
                if (at <= 0) continue;
                if (Enum.TryParse<SecondaryPart>(line[(at + 1)..].Trim(), true, out var part))
                    overrides[line[..at].Trim()] = part;
            }
            return overrides;
        }

        /// <summary>Per-kind physics on top of the file's (already bounded) values.</summary>
        public static StableSpringBoneSettings Shape(SecondaryClassification part, StableSpringBoneSettings s)
        {
            float stiffness = s.Stiffness, gravity = s.GravityPower, drag = s.Drag;
            switch (part.Kind)
            {
                case "ear":  // ears hold their shape
                    stiffness = Mathf.Max(stiffness, 3f); gravity = Mathf.Min(gravity, 0.02f); drag = Mathf.Max(drag, 0.5f);
                    break;
                case "tail":  // a fluffy tail: stiff enough to hold its shape when she walks
                    // (a spring joint lags behind a moving body, so stiffness, not drag, keeps it in)
                    stiffness = Mathf.Max(stiffness, 2f); gravity = Mathf.Max(gravity, 0.3f); drag = Mathf.Max(drag, 0.5f);
                    break;
                case "prop":  // carried things: heavy, swing little
                    stiffness = Mathf.Max(stiffness, 1.5f); gravity = Mathf.Max(gravity, 0.3f); drag = Mathf.Max(drag, 0.7f);
                    break;
                case "soft":
                    stiffness = Mathf.Max(stiffness, 1.5f); gravity = Mathf.Min(gravity, 0.05f); drag = Mathf.Max(drag, 0.55f);
                    break;
                default:
                    switch (part.Part)
                    {
                        case SecondaryPart.Hair:
                            // Real gravity. A spring chain trails a moving body by about
                            // drag ÷ stiffness, so walking doesn't stream long hair out behind her:
                            // stiffness is raised and drag left as authored (2026-10-15 live check).
                            stiffness = Mathf.Max(stiffness, 1.5f); gravity = Mathf.Max(gravity, 0.25f);
                            break;
                        case SecondaryPart.Clothing:  // heavier and more damped
                            stiffness = Mathf.Max(stiffness, 1f); gravity = Mathf.Max(gravity, 0.35f); drag = Mathf.Max(drag, 0.6f);
                            break;
                        default:
                            gravity = Mathf.Max(gravity, 0.08f); drag = Mathf.Max(drag, 0.45f);
                            break;
                    }
                    break;
            }
            return new StableSpringBoneSettings(stiffness, gravity, Vector3.down, drag, s.Radius);
        }

        /// <summary>
        /// Classify every spring chain and give chains without colliders ones generated from the
        /// body (sized from the avatar's own bones). Returns each joint's classification, for
        /// <see cref="Shape"/>. The caller rebuilds the spring simulation afterwards.
        /// </summary>
        public static Dictionary<VRM10SpringBoneJoint, SecondaryClassification> Prepare(
            Vrm10Instance instance, Animator animator, IReadOnlyDictionary<string, SecondaryPart> overrides,
            bool generateColliders = true)
        {
            var humanoid = new Dictionary<Transform, HumanBodyBones>();
            foreach (HumanBodyBones bone in Enum.GetValues(typeof(HumanBodyBones)))
            {
                if (bone == HumanBodyBones.LastBone) continue;
                var t = animator.GetBoneTransform(bone);
                if (t != null) humanoid[t] = bone;
            }
            RemoveDuplicateJoints(instance);
            var groups = generateColliders ? BodyColliders(animator) : new Dictionary<string, VRM10SpringBoneColliderGroup>();
            var result = new Dictionary<VRM10SpringBoneJoint, SecondaryClassification>();
            var counts = new Dictionary<string, int>();
            foreach (var spring in instance.SpringBone.Springs)
            {
                var joints = spring.Joints.Where(j => j != null).ToList();
                if (joints.Count == 0) continue;
                HumanBodyBones? anchor = null;
                for (var t = joints[0].transform.parent; t != null; t = t.parent)
                    if (humanoid.TryGetValue(t, out var bone)) { anchor = bone; break; }
                var root = joints[0].name;
                var part = overrides.TryGetValue(root, out var chosen)
                    ? new SecondaryClassification(chosen, chosen.ToString().ToLowerInvariant(), "override")
                    : Classify(joints.Select(j => j.name).Concat(new[] { spring.Name ?? "" }).ToList(), anchor);
                foreach (var joint in joints) result[joint] = part;
                var key = part.Part + "/" + part.Kind;
                counts[key] = counts.TryGetValue(key, out var n) ? n + 1 : 1;
                if (!root.Contains("_end")) Debug.Log($"Secondary part: {root} -> {part.Part} ({part.Kind}; {part.Reason})");
                if (spring.ColliderGroups.Count > 0 || groups.Count == 0) continue;
                // Generated colliders only where the author gave none.
                foreach (var name in part.Part switch
                {
                    // Not the arms: her posed arms move inside long hair and shove it every
                    // frame (owner report 2026-10-15: hair tips flapping).
                    SecondaryPart.Hair => new[] { "head", "torso" },
                    SecondaryPart.Clothing => new[] { "torso", "legs", "arms" },
                    _ => new[] { "head", "torso" },
                })
                    // A chain with any bone inside a body capsule in her rest pose (a tail inside
                    // the torso, hair lying on her back) would be shoved out of it every frame
                    // and never settle, so it doesn't collide with that region.
                    if (groups.TryGetValue(name, out var group) &&
                        !group.Colliders.Any(c => joints.Any(j => Inside(c, j.transform.position))))
                        spring.ColliderGroups.Add(group);
            }
            foreach (var group in groups.Values)
                if (!instance.SpringBone.ColliderGroups.Contains(group)) instance.SpringBone.ColliderGroups.Add(group);
            Debug.Log("Secondary parts: " + string.Join(", ", counts.OrderBy(c => c.Key).Select(c => $"{c.Key} {c.Value}")));
            return result;
        }

        /// <summary>
        /// Each bone is simulated by one spring only: the outermost chain that contains it. Some
        /// VRM 0.x files list a chain's root and then its descendants again as roots (Alice: 128 of
        /// 178 bones). UniVRM's 0.x migration already drops those duplicates (its "duplicated
        /// spring joint" warnings), so for Alice this removes nothing (checked 2026-10-15); it is a
        /// guard for files that reach the runtime with duplicates. Returns the number removed.
        /// </summary>
        public static int RemoveDuplicateJoints(Vrm10Instance instance)
        {
            static int Depth(Transform t) { var d = 0; for (var p = t.parent; p != null; p = p.parent) d++; return d; }
            var claimed = new HashSet<Transform>();
            var removed = 0;
            foreach (var spring in instance.SpringBone.Springs
                .Where(s => s.Joints.Any(j => j != null))
                .OrderBy(s => Depth(s.Joints.First(j => j != null).transform)).ToList())
            {
                var kept = new List<VRM10SpringBoneJoint>();
                foreach (var joint in spring.Joints)
                {
                    if (joint == null) continue;
                    if (claimed.Add(joint.transform)) kept.Add(joint); else removed++;
                }
                spring.Joints = kept;
            }
            instance.SpringBone.Springs.RemoveAll(s => s.Joints.Count == 0);
            if (removed > 0) Debug.Log($"Secondary parts: {removed} duplicated spring joints removed (each bone simulated once)");
            return removed;
        }

        /// <summary>Whether a world point is inside a capsule collider.</summary>
        public static bool Inside(VRM10SpringBoneCollider c, Vector3 point)
        {
            var t = c.transform;
            var scale = Mathf.Max(Mathf.Abs(t.lossyScale.x), Mathf.Max(Mathf.Abs(t.lossyScale.y), Mathf.Abs(t.lossyScale.z)));
            var start = t.TransformPoint(c.Offset);
            var end = c.ColliderType == VRM10SpringBoneColliderTypes.Capsule ? t.TransformPoint(c.Tail) : start;
            var axis = end - start;
            var along = axis.sqrMagnitude < 1e-12f ? 0f : Mathf.Clamp01(Vector3.Dot(point - start, axis) / axis.sqrMagnitude);
            return Vector3.Distance(point, start + axis * along) < c.Radius * scale;
        }

        /// <summary>Capsules on the body bones, from the avatar's own proportions, grouped by region.</summary>
        public static Dictionary<string, VRM10SpringBoneColliderGroup> BodyColliders(Animator animator)
        {
            var groups = new Dictionary<string, VRM10SpringBoneColliderGroup>();
            Transform? B(HumanBodyBones bone) => animator.GetBoneTransform(bone);
            var head = B(HumanBodyBones.Head);
            var hips = B(HumanBodyBones.Hips);
            var foot = B(HumanBodyBones.LeftFoot);
            if (head == null || hips == null || foot == null) return groups;
            var height = Mathf.Max(0.1f, (head.position.y - foot.position.y) * 1.1f);  // world units

            void Capsule(string group, HumanBodyBones bone, Vector3 start, Vector3 end, float radiusFraction)
            {
                var t = B(bone);
                if (t == null) return;
                if (!groups.TryGetValue(group, out var g))
                {
                    g = animator.gameObject.AddComponent<VRM10SpringBoneColliderGroup>();
                    g.Name = "SORA body " + group;
                    groups[group] = g;
                }
                var c = t.gameObject.AddComponent<VRM10SpringBoneCollider>();
                c.ColliderType = VRM10SpringBoneColliderTypes.Capsule;
                var scale = Mathf.Max(1e-6f, Mathf.Max(Mathf.Abs(t.lossyScale.x), Mathf.Max(Mathf.Abs(t.lossyScale.y), Mathf.Abs(t.lossyScale.z))));
                c.Offset = t.InverseTransformPoint(start);
                c.Tail = t.InverseTransformPoint(end);
                c.Radius = radiusFraction * height / scale;
                g.Colliders.Add(c);
            }
            Vector3 P(HumanBodyBones bone, Vector3 fallback) => B(bone)?.position ?? fallback;

            var neck = P(HumanBodyBones.Neck, head.position);
            var chest = P(HumanBodyBones.UpperChest, P(HumanBodyBones.Chest, (neck + hips.position) / 2));
            var up = Vector3.up * height;
            Capsule("head", HumanBodyBones.Head, head.position + up * 0.04f, head.position + up * 0.06f, 0.062f);
            Capsule("head", HumanBodyBones.Head, neck, head.position, 0.03f);
            Capsule("torso", HumanBodyBones.Hips, hips.position, chest, 0.07f);
            Capsule("torso", HumanBodyBones.Hips, chest, neck - up * 0.03f, 0.065f);
            foreach (var (upper, lower, end) in new[]
            {
                (HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftFoot),
                (HumanBodyBones.RightUpperLeg, HumanBodyBones.RightLowerLeg, HumanBodyBones.RightFoot),
            })
            {
                Capsule("legs", upper, P(upper, hips.position), P(lower, hips.position), 0.045f);
                Capsule("legs", lower, P(lower, hips.position), P(end, hips.position), 0.035f);
            }
            foreach (var (upper, lower, end) in new[]
            {
                (HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm, HumanBodyBones.LeftHand),
                (HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm, HumanBodyBones.RightHand),
            })
            {
                Capsule("arms", upper, P(upper, chest), P(lower, chest), 0.03f);
                Capsule("arms", lower, P(lower, chest), P(end, chest), 0.025f);
            }
            return groups;
        }
    }
}
