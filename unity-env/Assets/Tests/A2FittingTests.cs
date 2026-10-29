using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using MotionEnv.AvatarLayer;
using MotionEnv.Environment;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UniVRM10;
using UnityEngine;
using UnityEngine.TestTools;

/// <summary>Roadmap MA, A2 (fitting) and the shared foundations.</summary>
public sealed class A2FittingTests
{
    [Test]
    public void CanonicalTablesMatchTheSkeletonFile()
    {
        var json = JObject.Parse(File.ReadAllText(Path.Combine(TestAvatars.LatestVersion, "motionlab/assets/canonical_skeleton.json")));
        CollectionAssert.AreEqual(json["names"]!.Select(n => (string)n!).ToArray(), Canonical.Names);
        CollectionAssert.AreEqual(json["parents"]!.Select(n => (int)n!).ToArray(), Canonical.Parents);
        var maxAngle = json["max_angle_deg"]!.Select(n => n.Type == JTokenType.Null ? 0f : (float)n).ToArray();
        CollectionAssert.AreEqual(maxAngle, Canonical.MaxAngleDegrees);
        var offsets = json["offsets"]!.Select(o => o.Select(v => (float)v).ToArray()).ToArray();
        Assert.AreEqual(Canonical.HipsHeight, offsets[0][1], 1e-5f);
        var leg = new Vector3(offsets[15][0], offsets[15][1], offsets[15][2]).magnitude +
                  new Vector3(offsets[16][0], offsets[16][1], offsets[16][2]).magnitude;
        Assert.AreEqual(Canonical.LegLength, leg, 2e-3f);
    }

    [Test]
    public void TheAvatarMapAgreesWithPython()
    {
        var json = JObject.Parse(File.ReadAllText(Path.Combine(TestAvatars.LatestVersion, "tests/fixtures/adapter_cases.json")));
        var cases = 0;
        foreach (var c in json["cases"]!)
        {
            var missing = c["missing"]!.Select(n => Canonical.Index((string)n!)).ToHashSet();
            var map = AvatarMap.ForJoints(Enumerable.Range(0, Canonical.Count).Where(j => !missing.Contains(j)).ToList());
            CollectionAssert.AreEqual(c["joints"]!.Select(n => (string)n!).ToArray(), map.Joints.Select(j => Canonical.Names[j]).ToArray());
            var compose = c["compose"]!.Select(chain => string.Join(",", chain.Select(n => (string)n!))).ToArray();
            CollectionAssert.AreEqual(compose, map.Compose.Select(chain => string.Join(",", chain.Select(j => Canonical.Names[j]))).ToArray());
            CollectionAssert.AreEqual(c["dropped"]!.Select(n => (string)n!).ToArray(), map.Dropped.Select(j => Canonical.Names[j]).ToArray());
            foreach (var pose in c["poses"]!)
            {
                var canonical = pose["canonical"]!.Select(Matrix).ToArray();
                var expected = pose["avatar"]!.Select(Matrix).ToArray();
                var local = canonical.Select(m => m.rotation).ToArray();
                var output = new Quaternion[map.Joints.Count];
                map.ToAvatar(local, output);
                for (var i = 0; i < output.Length; i++)
                    Assert.Less(Quaternion.Angle(output[i], expected[i].rotation), 0.01f, $"{Canonical.Names[map.Joints[i]]}");
                cases++;
            }
        }
        Assert.AreEqual(21, cases);
    }

    private static Matrix4x4 Matrix(JToken rowMajor)
    {
        var v = rowMajor.Select(x => (float)x).ToArray();
        var m = Matrix4x4.identity;
        m.SetRow(0, new Vector4(v[0], v[1], v[2], 0));
        m.SetRow(1, new Vector4(v[3], v[4], v[5], 0));
        m.SetRow(2, new Vector4(v[6], v[7], v[8], 0));
        return m;
    }

    [Test]
    public void TheCopiesOfSorasCodeAreCurrent()
    {
        var sora = Path.Combine(TestAvatars.SoraRoot, "clients/unity/Assets/Scripts/Avatar");
        if (!Directory.Exists(sora)) Assert.Ignore("SORA isn't next to motion-lab");
        foreach (var name in new[] { "AvatarSecondaryParts.cs", "AvatarSecondaryMotionTuner.cs" })
        {
            using var sha = SHA256.Create();
            var mine = sha.ComputeHash(File.ReadAllBytes(Path.Combine(Application.dataPath, "FromSora", name)));
            var theirs = sha.ComputeHash(File.ReadAllBytes(Path.Combine(sora, name)));
            CollectionAssert.AreEqual(theirs, mine, name + " has drifted from SORA's; run unity-env/tools/sync-from-sora.ps1");
        }
    }

    /// <summary>Audit 2026-10-18: a perfectly straight leg has no bend plane; the fallback must
    /// bend the knee forward (the shin goes back), never backwards.</summary>
    [Test]
    public void IkBendsAStraightKneeForward()
    {
        var hip = new GameObject("hip").transform;
        var knee = new GameObject("knee").transform; knee.SetParent(hip, false); knee.localPosition = new Vector3(0, -0.4f, 0);
        var ankle = new GameObject("ankle").transform; ankle.SetParent(knee, false); ankle.localPosition = new Vector3(0, -0.4f, 0);
        TwoBoneIk.Solve(hip, knee, ankle, hip.position + new Vector3(0, -0.6f, 0), Vector3.forward);
        Debug.Log($"A2 IK: knee at {knee.position}, ankle at {ankle.position}");
        Assert.Greater(knee.position.z, 0.05f, "the knee points forward");
        Assert.Less(Vector3.Distance(ankle.position, hip.position + new Vector3(0, -0.6f, 0)), 1e-3f, "the ankle reaches the target");
        Object.Destroy(hip.gameObject);
    }

    /// <summary>Audit 2026-10-18: the layer assumes a T-pose rest; a VRM resting otherwise is
    /// reported (every pose would be offset by the difference).</summary>
    [UnityTest]
    public IEnumerator ARestPoseThatIsNotATPoseIsReported()
    {
        foreach (var avatar in TestAvatars.All.Where(TestAvatars.Available))
        {
            Vrm10Instance instance = null!;
            yield return TestAvatars.Load(avatar, i => instance = i);
            var fit = new AvatarFit(instance.GetComponent<Animator>());
            Assert.IsEmpty(fit.RestPoseIssues, avatar + " rests in a T-pose");
            Object.Destroy(instance.gameObject);
            yield return null;
        }
        Vrm10Instance aPosed = null!;
        yield return TestAvatars.Load("Seed-san", i => aPosed = i);
        var animator = aPosed.GetComponent<Animator>();
        animator.GetBoneTransform(HumanBodyBones.LeftUpperArm).rotation = Quaternion.Euler(0, 0, 40) * animator.GetBoneTransform(HumanBodyBones.LeftUpperArm).rotation;
        animator.GetBoneTransform(HumanBodyBones.RightUpperArm).rotation = Quaternion.Euler(0, 0, -40) * animator.GetBoneTransform(HumanBodyBones.RightUpperArm).rotation;
        var issues = new AvatarFit(animator).RestPoseIssues;
        Debug.Log("A2 A-pose: " + string.Join("; ", issues));
        Assert.That(issues.Count, Is.GreaterThanOrEqualTo(2), "both arms are reported");
        Assert.IsTrue(issues.Any(i => i.Contains("left upper arm")));
        Object.Destroy(aPosed.gameObject);
    }

    [UnityTest]
    public IEnumerator HipsTravelScalesWithLegLengthAndPlantedFeetStayPut()
    {
        foreach (var avatar in TestAvatars.All)
        {
            if (!TestAvatars.Available(avatar)) continue;
            Vrm10Instance instance = null!;
            yield return TestAvatars.Load(avatar, i => instance = i);
            var fit = new AvatarFit(instance.GetComponent<Animator>());
            Assert.IsEmpty(fit.RestPoseIssues);
            Debug.Log($"A2 {avatar}: leg {fit.LegLength:0.000} m, root scale {fit.RootScale:0.000}, height {fit.Height:0.00} m, joints {fit.Map.Joints.Count}");
            if (avatar == "Odd") Assert.Greater(fit.RootScale, 1.1f, "longer legs travel further");

            // A stand-in walk: the canonical root moves 1 m forward over 2 s while the gait swings.
            var motion = new StandInMotion();
            var pose = new CanonicalPose();
            var worst = 0f;
            for (var step = 0; step < 240; step++)
            {
                motion.Walk(step / 120f, 0.5f, pose);
                fit.Apply(pose);
                worst = Mathf.Max(worst, fit.PinnedDrift.Max());
            }
            var travelled = fit.Animator.GetBoneTransform(HumanBodyBones.Hips).position.z - fit.WorldSpace(fit.RestPosition(0)).z;
            Debug.Log($"A2 {avatar}: hips travelled {travelled:0.000} m for 1 canonical metre; worst planted-foot drift {worst * 1000:0.0} mm");
            Assert.AreEqual(1f * fit.RootScale, travelled, 0.01f * fit.RootScale, "hips travel scaled by leg length");
            Assert.Less(worst, 0.01f, "a planted foot stays within 1 cm");
            Object.Destroy(instance.gameObject);
            yield return null;
        }
    }

    /// <summary>The stand-in gait's stance leg: its IK puts the ankle where asked, so a planted
    /// foot stays on the floor through the whole stance, at walking and running speeds.</summary>
    [Test]
    public void StanceLegIkKeepsThePlantedFootOnTheFloor()
    {
        const float thigh = 0.370993f, shin = 0.385462f;
        static Vector2 Ankle(float hip, float knee)
        {
            float h = hip * Mathf.Deg2Rad, s = (hip - knee) * Mathf.Deg2Rad;
            return new Vector2(thigh * Mathf.Sin(h) + shin * Mathf.Sin(s), thigh * Mathf.Cos(h) + shin * Mathf.Cos(s));
        }
        foreach (var (forward, down) in new[] { (0f, 0.70f), (0.3f, 0.65f), (-0.35f, 0.62f), (0.1f, 0.5f) })
        {
            var (hip, knee) = StandInMotion.LegIk(forward, down);
            var ankle = Ankle(hip, knee);
            Assert.AreEqual(forward, ankle.x, 1e-4f, $"forward, target ({forward}, {down})");
            Assert.AreEqual(down, ankle.y, 1e-4f, $"down, target ({forward}, {down})");
            Assert.GreaterOrEqual(knee, 0f, "the knee bends forwards only");
        }
        var motion = new StandInMotion();
        var pose = new CanonicalPose();
        // The gait's leg rotations are about X only.
        float X(string joint)
        {
            var q = pose.Local[Canonical.Index(joint)];
            return 2f * Mathf.Atan2(q.x, q.w) * Mathf.Rad2Deg;
        }
        foreach (var speed in new[] { 0.5f, 1.2f, 2.4f, 3.5f })
        {
            var heights = new System.Collections.Generic.List<float>();
            for (var i = 0; i < 200; i++)
            {
                motion.Gait(i / 200f, speed, pose);
                if (!pose.Contacts[0]) continue;
                var hip = -X("leftUpperLeg");
                var knee = X("leftLowerLeg");
                var foot = X("leftFoot");
                // Until the heel lifts, the ankle sits at one height and the foot is level.
                if (foot - (hip - knee) > 0.01f) continue;
                heights.Add(pose.Root.y - Ankle(hip, knee).y);
                Assert.AreEqual(0f, foot - (hip - knee), 0.01f, "foot level");
                Assert.Less(knee, 75f, $"stance knee at {speed} m/s");
            }
            Assert.Greater(heights.Count, 10, $"stance samples at {speed} m/s");
            Assert.Less(heights.Max() - heights.Min(), 0.002f, $"planted ankle height varies at {speed} m/s");
        }
    }
}
