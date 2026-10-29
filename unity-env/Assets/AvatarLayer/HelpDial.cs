using System;

namespace MotionEnv.AvatarLayer
{
    /// <summary>
    /// Everything non-physical, as settings from 0 to 1 that default to zero (owner, 2026-10-15:
    /// "Ideally I'd like for the model to be singularly subject to the physics constraints").
    /// Every use is logged in <see cref="HelpLog"/>, so each scenario's least help is measurable.
    /// </summary>
    [Serializable]
    public struct HelpDial
    {
        /// <summary>Joints exempt from physics: at 1 every joint follows its target almost
        /// rigidly, whatever the forces.</summary>
        public float Exempt;
        /// <summary>Motor strength above realistic muscle limits: 0 is realistic, 1 is ten times.</summary>
        public float Strength;
        /// <summary>A balance helper at the hips: forces and torques pulling the hips towards
        /// where the motion wants them, like a puppet string.</summary>
        public float Balance;

        public HelpDial(float exempt, float strength, float balance)
        {
            Exempt = Clamp(exempt);
            Strength = Clamp(strength);
            Balance = Clamp(balance);
        }

        public bool Any => Exempt > 0 || Strength > 0 || Balance > 0;

        /// <summary>The physics ladder's four steps, from full help to none.</summary>
        public static readonly (string Name, HelpDial Dial)[] Ladder =
        {
            ("1 full help", new HelpDial(1, 1, 1)),
            ("2 no exempt joints", new HelpDial(0, 1, 1)),
            ("3 realistic motor strength", new HelpDial(0, 0, 1)),
            ("4 no balance helper", new HelpDial(0, 0, 0)),
        };

        public static HelpDial None => default;

        public override string ToString() => $"exempt {Exempt:0.##}, strength {Strength:0.##}, balance {Balance:0.##}";

        private static float Clamp(float value) => float.IsNaN(value) ? 0 : Math.Clamp(value, 0f, 1f);
    }

    /// <summary>How much help was actually used.</summary>
    public sealed class HelpLog
    {
        /// <summary>Seconds with any joint exempt.</summary>
        public double ExemptSeconds;
        /// <summary>Seconds with motors stronger than realistic.</summary>
        public double StrengthSeconds;
        /// <summary>Total balance-helper impulse: linear (N·s) and angular (N·m·s).</summary>
        public double BalanceImpulse, BalanceAngularImpulse;
        /// <summary>The largest balance-helper force (N) in a step.</summary>
        public double PeakBalanceForce;

        public bool Used => ExemptSeconds > 0 || StrengthSeconds > 0 || BalanceImpulse > 0 || BalanceAngularImpulse > 0;

        public void Reset()
        {
            ExemptSeconds = StrengthSeconds = BalanceImpulse = BalanceAngularImpulse = PeakBalanceForce = 0;
        }
    }
}
