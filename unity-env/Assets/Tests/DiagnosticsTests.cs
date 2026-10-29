using System.Collections.Generic;
using System.Collections;
using System.Linq;
using System.Text;
using MotionEnv.AvatarLayer;
using MotionEnv.Environment;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

/// <summary>Diagnostics, not gates: run by name only (Explicit).</summary>
public sealed class DiagnosticsTests
{
    /// <summary>Traces the first 1.5 s of the run at full help: hips tilt and its direction,
    /// hips speed against the wanted speed, the balance helper's pull, and what touches the floor.</summary>
    [UnityTest, Explicit, Timeout(1800000)]
    public IEnumerator RunTrace()
    {
        foreach (var avatar in TestAvatars.All.Where(TestAvatars.Available))
        {
            var log = new StringBuilder($"run trace [{avatar}, full help]").AppendLine();
            var runner = new ScenarioRunner { Seconds = 1.5f, CheckClipping = false };
            var step = 0;
            runner.Trace = (t, body, falls, surfaces, pose) =>
            {
                if (step++ % 6 != 0 || t < 0.4f) return;
                var hips = body.Links[0];
                var up = hips.transform.up;
                var tilt = Vector3.Angle(up, Vector3.up);
                var touching = string.Join(",", body.Colliders.Where(c => surfaces.Contact(c).Gap < 0.003f)
                    .Select(c => c.name.Replace("Link ", "")).Distinct());
                var wantedV = ScenarioRunner.RunSpeed * Mathf.Clamp01(t);
                log.AppendLine($"{t:0.00} {falls.Phase} tilt {tilt:0} toward ({up.x:0.00},{up.z:0.00}) hipsV ({hips.linearVelocity.x:0.00},{hips.linearVelocity.y:0.00},{hips.linearVelocity.z:0.00}) wanted z {wantedV:0.00} hipsY {hips.transform.position.y:0.000} contacts L{(pose.Contacts[0] ? 1 : 0)}R{(pose.Contacts[2] ? 1 : 0)} touching {touching}");
            };
            yield return runner.Run(avatar, "run", "1 full help", HelpDial.Ladder[0].Dial, _ => { });
            Debug.Log(log.ToString());
        }
    }
}
