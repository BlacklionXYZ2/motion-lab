using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace MotionEnv.AvatarLayer
{
    /// <summary>
    /// The C# twin of <c>motionlab.adapter.AvatarMap</c>: the model's 22 canonical joints onto an
    /// avatar's own joints. A missing leaf is dropped; a missing only child folds into its parent
    /// (<c>chest ← chest · upperChest</c>); a missing joint with siblings and one child folds into
    /// that child (<c>head ← neck · head</c>). Every joint the avatar has keeps the exact world
    /// orientation the model meant. Shared cases from Python check the two agree.
    /// </summary>
    public sealed class AvatarMap
    {
        private static readonly List<int>[] Children = BuildChildren();

        /// <summary>Canonical indices of the avatar's joints, in canonical order.</summary>
        public IReadOnlyList<int> Joints { get; }
        /// <summary>For each avatar joint, the canonical joints multiplied into it, in order.</summary>
        public IReadOnlyList<int[]> Compose { get; }
        public IReadOnlyList<int> Dropped { get; }

        private AvatarMap(List<int> joints, List<int[]> compose, List<int> dropped)
        {
            Joints = joints;
            Compose = compose;
            Dropped = dropped;
        }

        private static List<int>[] BuildChildren()
        {
            var children = Enumerable.Range(0, Canonical.Count).Select(_ => new List<int>()).ToArray();
            for (var j = 1; j < Canonical.Count; j++) children[Canonical.Parents[j]].Add(j);
            return children;
        }

        public static AvatarMap ForJoints(ICollection<int> present)
        {
            var missingRequired = Enumerable.Range(0, Canonical.Count)
                .Where(j => !present.Contains(j) && !Canonical.Optional.Contains(Canonical.Names[j])).ToList();
            if (missingRequired.Count > 0)
                throw new ArgumentException("an avatar must have the VRM's required joints: " +
                    string.Join(", ", missingRequired.Select(j => Canonical.Names[j])));
            var lists = new Dictionary<int, List<int>>();
            foreach (var j in present) lists[j] = new List<int> { j };
            var pending = new Dictionary<int, List<int>>(); // child → missing joints in front of it
            var target = new Dictionary<int, int>();
            var dropped = new List<int>();
            for (var name = 0; name < Canonical.Count; name++)
            {
                var parent = Canonical.Parents[name];
                if (present.Contains(name))
                {
                    target[name] = name;
                    if (pending.Remove(name, out var front)) lists[name].InsertRange(0, front);
                    continue;
                }
                var children = Children[name];
                if (children.Count == 0) dropped.Add(name);
                else if (parent >= 0 && Children[parent].Count == 1 && target.ContainsKey(parent))
                {
                    var owner = target[parent];
                    lists[owner].Add(name);
                    target[name] = owner;
                }
                else if (children.Count == 1)
                {
                    var front = pending.Remove(name, out var earlier) ? earlier : new List<int>();
                    front.Add(name);
                    pending[children[0]] = front;
                }
                else throw new ArgumentException("can't fold away " + Canonical.Names[name] + ": it has several children");
            }
            var joints = Enumerable.Range(0, Canonical.Count).Where(present.Contains).ToList();
            return new AvatarMap(joints, joints.Select(j => lists[j].ToArray()).ToList(), dropped);
        }

        /// <summary>The nearest present ancestor of an avatar joint, as a canonical index, or −1.</summary>
        public int PresentParent(int joint)
        {
            var parent = Canonical.Parents[joint];
            while (parent >= 0 && !Joints.Contains(parent)) parent = Canonical.Parents[parent];
            return parent;
        }

        /// <summary>Canonical local rotations → the avatar's local rotations (same frames).</summary>
        public void ToAvatar(Quaternion[] canonicalLocal, Quaternion[] avatarLocal)
        {
            for (var i = 0; i < Compose.Count; i++)
            {
                var chain = Compose[i];
                var r = canonicalLocal[chain[0]];
                for (var k = 1; k < chain.Length; k++) r *= canonicalLocal[chain[k]];
                avatarLocal[i] = r;
            }
        }
    }
}
