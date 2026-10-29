using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using MotionEnv.AvatarLayer;
using MotionEnv.Environment;
using Newtonsoft.Json;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

/// <summary>Roadmap MA: A4 (falling), A1's clipping checks, and the physics ladder.</summary>
public sealed class A4LadderTests
{
    /// <summary>The falling gate: after a window closes under her, she falls to the taskbar and
    /// comes to rest within 2 s of impact, at most 1 cm above it and 1 cm into it.</summary>
    [UnityTest]
    [Timeout(1800000)]
    public IEnumerator SheFallsAndComesToRestOnTheTaskbar()
    {
        var runner = new ScenarioRunner { Seconds = 8f, CheckClipping = false };
        foreach (var avatar in TestAvatars.All.Where(TestAvatars.Available))
        {
            ScenarioResult result = null!;
            yield return runner.Run(avatar, "fall", "no help", HelpDial.None, r => result = r);
            Debug.Log("A4 " + result + " | events: " + string.Join("; ", result.Events));
            // What is seen: her mesh, not just the collision shapes, against the taskbar.
            Assert.IsTrue(result.Metrics.ContainsKey("mesh lowest above floor cm"), "the mesh was measured at rest");
            // Her collision shapes are fitted to her mesh: at rest her body (not hair or cloth on
            // spring joints) is at most a few centimetres into the floor, and not hovering.
            Assert.Less(result.Metrics["mesh deepest into floor cm"], 7.0, "body mesh into the floor");
            Assert.Less(result.Metrics["mesh lowest above floor cm"], 2.0, "body mesh hovering");
            Assert.IsTrue(result.Passed, result.ToString());
        }
    }

    /// <summary>Falls settle within the gate whatever help she had before falling (help is off
    /// during the fall, so only her state beforehand differs).</summary>
    [UnityTest]
    [Timeout(3600000)]
    public IEnumerator FallsSettleAtEveryHelpStep()
    {
        var runner = new ScenarioRunner { Seconds = 9f, CheckClipping = false };
        var failures = new List<string>();
        foreach (var avatar in TestAvatars.All.Where(TestAvatars.Available))
        foreach (var (step, dial) in HelpDial.Ladder)
        {
            ScenarioResult result = null!;
            yield return runner.Run(avatar, "fall", step, dial, r => result = r);
            Debug.Log($"A4 fall [{avatar}, {step}]: {(result.Passed ? "pass" : "FAIL")} ({result.Reason}), mesh {result.Metrics.GetValueOrDefault("mesh deepest into floor cm", -1):0.0} cm; {string.Join("; ", result.Events.Skip(Math.Max(0, result.Events.Count - (result.Passed ? 2 : 6))))}");
            if (!result.Passed) failures.Add(result.ToString());
        }
        Assert.IsEmpty(failures);
    }

    /// <summary>
    /// The physics ladder (roadmap MA): every scenario on every test avatar, at each help step.
    /// The least help each scenario needs is recorded; with full help everything must pass, so
    /// the environment itself is sound. Writes <c>outputs/ma/ladder.json</c> and <c>ladder.md</c>.
    /// </summary>
    [UnityTest]
    [Category("Ladder")]
    [Timeout(7200000)]
    public IEnumerator PhysicsLadder()
    {
        var results = new List<ScenarioResult>();
        var runner = new ScenarioRunner();
        foreach (var avatar in TestAvatars.All.Where(TestAvatars.Available))
        foreach (var scenario in ScenarioRunner.Names)
        foreach (var (step, dial) in HelpDial.Ladder)
        {
            runner.CheckClipping = step == HelpDial.Ladder[0].Name; // hair and clothing, measured once with full help
            ScenarioResult result = null!;
            yield return runner.Run(avatar, scenario, step, dial, r => result = r);
            Debug.Log("MA ladder " + result);
            results.Add(result);
        }
        Write(results);
        var clipped = results.Where(r => r.ClippingPassed == false).Select(r => $"{r.Avatar} / {r.Scenario}: {r.ClippingNote}").ToList();
        var fullHelpFailures = results.Where(r => r.Step == HelpDial.Ladder[0].Name && !r.Passed).Select(r => r.ToString()).ToList();
        Assert.IsEmpty(fullHelpFailures.Concat(clipped), "with full help every scenario passes, and hair and clothing stay out of the body");
    }

    private static void Write(List<ScenarioResult> results)
    {
        var folder = TestAvatars.OutputFolder("ma");
        var rows = results.Select(r => new
        {
            avatar = r.Avatar, scenario = r.Scenario, step = r.Step, passed = r.Passed, reason = r.Reason,
            help = new { exempt = r.Help.Exempt, strength = r.Help.Strength, balance = r.Help.Balance },
            clipping = r.ClippingPassed == null ? null : new { passed = r.ClippingPassed, note = r.ClippingNote },
            events = r.Events,
            help_used = new { exempt_s = r.HelpUsed.ExemptSeconds, strength_s = r.HelpUsed.StrengthSeconds,
                balance_impulse_ns = r.HelpUsed.BalanceImpulse, balance_angular_impulse_nms = r.HelpUsed.BalanceAngularImpulse,
                peak_balance_force_n = r.HelpUsed.PeakBalanceForce },
            metrics = r.Metrics,
        });
        File.WriteAllText(Path.Combine(folder, "ladder.json"), JsonConvert.SerializeObject(new
        {
            written = System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss zzz"),
            motion = "stand-in (hand-made poses; M1's model replaces them)",
            physics = "120 Hz, TGS solver, avatars at their own size in metres",
            runs = rows,
        }, Formatting.Indented));

        var md = new StringBuilder();
        md.AppendLine("# Physics ladder").AppendLine();
        md.AppendLine($"Written {System.DateTime.Now:yyyy-MM-dd HH:mm} by the Unity environment (`A4LadderTests.PhysicsLadder`). Motion: the stand-in poses, not a trained model.").AppendLine();
        md.AppendLine("## Least help each scenario needs").AppendLine();
        md.AppendLine("The last ladder step that passes, removing help in order: 1 full help, 2 no exempt joints, 3 realistic motor strength, 4 no balance helper.").AppendLine();
        var avatars = results.Select(r => r.Avatar).Distinct().ToList();
        md.AppendLine("| Scenario | " + string.Join(" | ", avatars) + " |");
        md.AppendLine("|---|" + string.Concat(avatars.Select(_ => "---|")));
        foreach (var scenario in ScenarioRunner.Names)
        {
            var cells = avatars.Select(a =>
            {
                var least = "none passes";
                foreach (var (step, _) in HelpDial.Ladder)
                {
                    var r = results.FirstOrDefault(x => x.Avatar == a && x.Scenario == scenario && x.Step == step);
                    if (r == null || !r.Passed) break;
                    least = step;
                }
                return least;
            });
            md.AppendLine($"| {scenario} | " + string.Join(" | ", cells) + " |");
        }
        md.AppendLine().AppendLine("## Hair, clothing and accessories (A1)").AppendLine();
        md.AppendLine($"Deepest a joint went into the body colliders its chain collides with, during each scenario at full help (allowed {ClippingProbe.Allowed * 100:0} cm).").AppendLine();
        md.AppendLine("| Scenario | " + string.Join(" | ", avatars) + " |");
        md.AppendLine("|---|" + string.Concat(avatars.Select(_ => "---|")));
        foreach (var scenario in ScenarioRunner.Names)
            md.AppendLine($"| {scenario} | " + string.Join(" | ", avatars.Select(a =>
                results.FirstOrDefault(x => x.Avatar == a && x.Scenario == scenario && x.ClippingPassed != null) is { } r
                    ? (r.ClippingPassed == true ? "" : "**over:** ") + r.ClippingNote : "not run")) + " |");
        md.AppendLine().AppendLine("## Every run").AppendLine();
        foreach (var r in results) md.AppendLine("- " + r);
        File.WriteAllText(Path.Combine(folder, "ladder.md"), md.ToString());
    }
}
