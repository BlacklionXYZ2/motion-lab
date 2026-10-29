using UnityEngine;

namespace MotionEnv.AvatarLayer
{
    /// <summary>
    /// Hand-made canonical poses that drive the environment until the motion core exists: stand,
    /// walk, sit, and the fall reaction and get-up poses. M1's model replaces them through the
    /// same <see cref="CanonicalPose"/> interface; nothing downstream changes.
    /// <para>Signs (Unity frame, world-aligned joint frames): +X bends a knee or leans the spine
    /// forward, −X lifts a thigh forward, +Z lowers the left arm (−Z the right), −X swings a
    /// lowered arm forward, +Y bends the left elbow (−Y the right).</para>
    /// </summary>
    public sealed class StandInMotion
    {
        public const float ArmsDown = 72f;
        /// <summary>One walking cycle (both steps) at an ordinary walk, seconds.</summary>
        public const float WalkPeriod = 1.1f;

        public void Stand(float t, CanonicalPose pose)
        {
            Clear(pose);
            ArmsAtSides(pose, 0, 0);
            var breathe = Mathf.Sin(2 * Mathf.PI * 0.25f * t);
            pose.Set("spine", new Vector3(1f + 0.6f * breathe, 0, 0));
            pose.Set("leftUpperLeg", new Vector3(-3, 0, 0));
            pose.Set("rightUpperLeg", new Vector3(-3, 0, 0));
            pose.Set("leftLowerLeg", new Vector3(5, 0, 0));
            pose.Set("rightLowerLeg", new Vector3(5, 0, 0));
            pose.Set("leftFoot", new Vector3(-2, 0, 0));
            pose.Set("rightFoot", new Vector3(-2, 0, 0));
            pose.Root = new Vector3(0, Canonical.HipsHeight - 0.004f, 0);
            for (var c = 0; c < 4; c++) pose.Contacts[c] = true;
        }

        /// <summary>A walk or run at a constant speed in canonical metres a second; the root moves
        /// forward (+Z) at that speed.</summary>
        public void Walk(float t, float speed, CanonicalPose pose)
        {
            Gait(t / GaitPeriod(speed), speed, pose);
            pose.Root = new Vector3(0, pose.Root.y, speed * t);
        }

        /// <summary>One gait cycle (both steps), seconds: quicker as she speeds up (about 1.8
        /// steps a second walking, 3.6 sprinting).</summary>
        public static float GaitPeriod(float speed) => Mathf.Clamp(1.1f - 0.2f * (speed - 0.5f), 0.55f, 1.1f);

        /// <summary>The share of a cycle each foot is on the ground: 0.6 walking (both feet down
        /// between steps), falling to 0.3 running at 2.4 m/s and faster (a moment with both feet
        /// in the air). A short contact keeps the stance stride short enough to land on a
        /// slightly bent knee.</summary>
        public static float DutyFactor(float speed) => Mathf.Lerp(0.6f, 0.3f, Mathf.InverseLerp(1.2f, 2.4f, speed));

        // Canonical leg (canonical_skeleton.json): hip to knee, knee to ankle, ankle to the ball
        // of the foot; the hip joint's drop below the hips; the ankle's height standing.
        private const float Thigh = 0.370993f, Shin = 0.385462f, FootToBall = 0.118356f;
        private const float HipJointDrop = 0.047871f;
        private const float AnkleHeight = Canonical.HipsHeight - 0.004f - HipJointDrop - 0.370351f - 0.38421f;
        /// <summary>The furthest a stance leg straightens, as a share of its full length.</summary>
        private const float StanceReach = 0.985f;

        /// <summary>Two-bone leg IK in the sagittal plane: hip flexion and knee bend (degrees,
        /// the stand-in's signs: positive lifts the thigh forward or bends the knee) that put the
        /// ankle <paramref name="forward"/> ahead of and <paramref name="down"/> below the hip
        /// joint. Out of reach, the leg points straight at the target.</summary>
        public static (float Hip, float Knee) LegIk(float forward, float down)
        {
            var d = Mathf.Clamp(Mathf.Sqrt(forward * forward + down * down), 1e-4f, (Thigh + Shin) * 0.9999f);
            var knee = Mathf.PI - Mathf.Acos(Mathf.Clamp((Thigh * Thigh + Shin * Shin - d * d) / (2 * Thigh * Shin), -1f, 1f));
            var towardTarget = Mathf.Atan2(forward, down);
            var thighAhead = Mathf.Acos(Mathf.Clamp((Thigh * Thigh + d * d - Shin * Shin) / (2 * Thigh * d), -1f, 1f));
            return ((towardTarget + thighAhead) * Mathf.Rad2Deg, knee * Mathf.Rad2Deg);
        }

        /// <summary>How far through its swing a leg is (0 on the ground, 1 mid-swing).</summary>
        public static float SwingAmount(float legPhase, float duty)
        {
            var p = Mathf.Repeat(legPhase, 1f);
            return p < duty ? 0f : Mathf.Sin(Mathf.PI * (p - duty) / (1f - duty));
        }

        /// <summary>Whether this speed's gait has a moment with both feet off the ground.</summary>
        public static bool HasFlight(float speed) => DutyFactor(speed) < 0.5f;

        /// <summary>
        /// A walk or run at a point in its cycle (<paramref name="phase"/> counts cycles; the
        /// caller advances it by <c>dt / GaitPeriod(speed)</c>, so speed can change smoothly).
        /// Each foot is planted from in front of the hips to behind them while the hips travel
        /// <c>speed × time on the ground</c>, so the stride matches the speed. The root's forward
        /// position is the caller's.
        /// </summary>
        public void Gait(float phase, float speed, CanonicalPose pose)
        {
            Clear(pose);
            var period = GaitPeriod(speed);
            var duty = DutyFactor(speed);
            var run = Mathf.InverseLerp(1.2f, 2.4f, speed);
            var sweep = speed * duty * period;                    // how far the hips pass over a planted foot
            // Hips sit lower than standing, low enough that a planted foot stays in reach at both
            // ends of its stride, and bob twice a cycle.
            var bobSize = Mathf.Lerp(0.008f, 0.02f, run);
            var reachable = Mathf.Sqrt(Mathf.Max(0f, Sq(StanceReach * (Thigh + Shin)) - Sq(sweep / 2f))) + HipJointDrop + AnkleHeight - bobSize;
            var hipsY = Mathf.Min(Canonical.HipsHeight - Mathf.Lerp(0.035f, 0.06f, run), reachable)
                        + bobSize * Mathf.Cos(4f * Mathf.PI * phase);
            var down = hipsY - HipJointDrop - AnkleHeight;
            // Heel lift at toe-off (degrees) comes from pushing off forwards: none stepping on the
            // spot, where it would leave her on tiptoe on one foot.
            var push = Mathf.Lerp(8f, 20f, run) * Mathf.InverseLerp(0f, 0.4f, speed);
            // A stance leg at stance progress s: the ankle planted on the floor, from half a sweep
            // in front of the hips to half behind, the heel lifting at the end (foot pivoting on
            // its ball). The foot stays level until then.
            (float Hip, float Knee, float Foot) Stance(float s)
            {
                var heel = push * Mathf.SmoothStep(0f, 1f, (s - 0.6f) / 0.4f);
                var (hip, knee) = LegIk(sweep * (0.5f - s), down - FootToBall * Mathf.Sin(heel * Mathf.Deg2Rad));
                return (hip, knee, hip - knee + heel);
            }
            var touchdown = Stance(0f);
            var toeOff = Stance(1f);
            var amplitude = Mathf.Max(1f, touchdown.Hip);
            float flexLeft = 0f, flexRight = 0f;
            for (var leg = 0; leg < 2; leg++)
            {
                var p = Mathf.Repeat(phase + leg * 0.5f, 1f);
                var side = leg == 0 ? "left" : "right";
                float hip, knee, foot;
                if (p < duty) (hip, knee, foot) = Stance(p / duty);
                else
                {
                    // In the air: from toe-off back through to touchdown, the thigh lifted, the
                    // knee folded and the toes up for clearance.
                    var s = (p - duty) / (1f - duty);
                    var lift = Mathf.Sin(Mathf.PI * s);
                    var through = Mathf.SmoothStep(0f, 1f, s);
                    hip = Mathf.Lerp(toeOff.Hip, touchdown.Hip, through) + Mathf.Lerp(18f, 40f, run) * lift;
                    knee = Mathf.Lerp(toeOff.Knee, touchdown.Knee, through) + Mathf.Lerp(55f, 105f, run) * Mathf.Sin(Mathf.PI * Mathf.Pow(s, 0.8f));
                    foot = Mathf.Lerp(toeOff.Foot, touchdown.Foot, through) - 12f * lift;
                }
                if (leg == 0) flexLeft = hip; else flexRight = hip;
                pose.Set(side + "UpperLeg", new Vector3(-hip, 0, 0));  // −X lifts the thigh forward
                pose.Set(side + "LowerLeg", new Vector3(knee, 0, 0));
                pose.Set(side + "Foot", new Vector3(foot, 0, 0));
                pose.Contacts[leg * 2] = pose.Contacts[leg * 2 + 1] = p < duty;
            }
            // Arms swing against the legs, more and with bent elbows when running.
            var arm = Mathf.Lerp(0.9f, 1.2f, run);
            ArmsAtSides(pose, -arm * flexRight, -arm * flexLeft, elbow: Mathf.Lerp(10f, 80f, run));
            pose.Set("spine", new Vector3(Mathf.Lerp(3f, 10f, run), 2f * (flexLeft - flexRight) / amplitude, 0));
            pose.Root = new Vector3(0, hipsY, 0);
        }

        private static float Sq(float x) => x * x;

        /// <summary>Sitting on a seat at about knee height behind her; blend 0 (standing) to 1.</summary>
        public void Sit(float blend, CanonicalPose pose)
        {
            var s = Mathf.SmoothStep(0, 1, blend);
            Stand(0, pose);
            pose.Set("leftUpperLeg", new Vector3(-88 * s - 3 * (1 - s), 0, 0));
            pose.Set("rightUpperLeg", new Vector3(-88 * s - 3 * (1 - s), 0, 0));
            pose.Set("leftLowerLeg", new Vector3(88 * s + 5 * (1 - s), 0, 0));
            pose.Set("rightLowerLeg", new Vector3(88 * s + 5 * (1 - s), 0, 0));
            pose.Set("spine", new Vector3(8 * s, 0, 0));
            ArmsAtSides(pose, -25 * s, -25 * s, elbow: 40 * s);
            pose.Root = new Vector3(0, Mathf.Lerp(Canonical.HipsHeight, 0.47f, s), Mathf.Lerp(0, -0.33f, s));
        }

        /// <summary>
        /// The reaction while falling (A4 step 2): arms out towards the fall, chin tucked, a
        /// catch step and a turn towards the side. <paramref name="direction"/> is the horizontal
        /// fall direction in her own frame (x right, z forward).
        /// </summary>
        public void FallReaction(Vector2 direction, CanonicalPose pose)
        {
            Clear(pose);
            var d = direction.sqrMagnitude > 1e-4f ? direction.normalized : new Vector2(0, -1);
            var forward = d.y;   // +1 falling forwards, −1 backwards
            var side = d.x;      // +1 towards her right
            // Arms reach towards the fall, forward or back, with soft elbows.
            var reach = forward >= 0 ? -70f : 40f;
            ArmsAtSides(pose, reach, reach, elbow: 25, spread: 30 + 25 * Mathf.Abs(side));
            pose.Set("neck", new Vector3(25 * Mathf.Max(0, -forward) + 10, 0, 0)); // chin tucked
            pose.Set("head", new Vector3(10, 0, -15 * side));
            pose.Set("spine", new Vector3(forward >= 0 ? 15 : 25, 20 * side, 0));    // curl and turn
            // A catch step with the leg on the falling side, the other knee giving way.
            var stepLeft = side < 0 || (Mathf.Abs(side) < 0.3f);
            var stepHip = forward >= 0 ? -45f : -20f;
            pose.Set(stepLeft ? "leftUpperLeg" : "rightUpperLeg", new Vector3(stepHip, 0, 0));
            pose.Set(stepLeft ? "leftLowerLeg" : "rightLowerLeg", new Vector3(40, 0, 0));
            pose.Set(stepLeft ? "rightLowerLeg" : "leftLowerLeg", new Vector3(55, 0, 0));
            pose.Set(stepLeft ? "rightUpperLeg" : "leftUpperLeg", new Vector3(-25, 0, 0));
        }

        /// <summary>A loose, slightly curled pose for the settling phase.</summary>
        public void Relaxed(CanonicalPose pose)
        {
            Clear(pose);
            ArmsAtSides(pose, -20, -20, elbow: 30, spread: 10);
            pose.Set("spine", new Vector3(10, 0, 0));
            pose.Set("neck", new Vector3(10, 0, 0));
            foreach (var leg in new[] { "leftUpperLeg", "rightUpperLeg" }) pose.Set(leg, new Vector3(-25, 0, 0));
            foreach (var knee in new[] { "leftLowerLeg", "rightLowerLeg" }) pose.Set(knee, new Vector3(35, 0, 0));
        }

        /// <summary>Kneeling with hands on the ground: the get-up stand-in's middle pose.</summary>
        public void Kneel(CanonicalPose pose)
        {
            Clear(pose);
            ArmsAtSides(pose, -60, -60, elbow: 10);
            pose.Set("spine", new Vector3(35, 0, 0));
            pose.Set("leftUpperLeg", new Vector3(-95, 0, 0));
            pose.Set("leftLowerLeg", new Vector3(100, 0, 0));
            pose.Set("rightUpperLeg", new Vector3(-30, 0, 0));
            pose.Set("rightLowerLeg", new Vector3(120, 0, 0));
            pose.Set("rightFoot", new Vector3(30, 0, 0));
            pose.Root = new Vector3(0, 0.45f, 0);
        }

        private static void ArmsAtSides(CanonicalPose pose, float leftSwing, float rightSwing, float elbow = 8, float spread = 0)
        {
            // Lower each arm from the T-pose, then swing it forward (−) or back (+) about X.
            pose.Local[Canonical.Index("leftUpperArm")] = Quaternion.Euler(leftSwing, 0, 0) * Quaternion.Euler(0, 0, ArmsDown - spread);
            pose.Local[Canonical.Index("rightUpperArm")] = Quaternion.Euler(rightSwing, 0, 0) * Quaternion.Euler(0, 0, -(ArmsDown - spread));
            pose.Set("leftLowerArm", new Vector3(0, elbow, 0));
            pose.Set("rightLowerArm", new Vector3(0, -elbow, 0));
        }

        private static void Clear(CanonicalPose pose)
        {
            for (var j = 0; j < Canonical.Count; j++) pose.Local[j] = Quaternion.identity;
            pose.Root = new Vector3(0, Canonical.HipsHeight, 0);
            for (var c = 0; c < 4; c++) pose.Contacts[c] = false;
        }
    }
}
