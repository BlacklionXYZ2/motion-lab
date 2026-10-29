using System;
using System.Linq;
using UnityEngine;

namespace MotionEnv.AvatarLayer
{
    public enum FallPhase { Supported, Reacting, Settling, Rested, GettingUp }

    /// <summary>
    /// A4, falling: physical, not a preset animation (owner, 2026-10-15).
    /// <list type="number">
    /// <item><b>Trigger:</b> losing support, a push that tips her past recovery, or being dropped.
    /// The direction comes from her momentum.</item>
    /// <item><b>Reaction</b> (0.35 s, motors partly on): arms out towards the fall, chin tucked, a
    /// catch step and a turn to the side. Stand-in poses until M1 learns them.</item>
    /// <item><b>Impact and settling:</b> muscle tone drops to 15% (the neck keeps 40%); she comes
    /// to rest in real contact with whatever she landed on.</item>
    /// <item><b>Getting up</b> from the pose she ended in. The stand-in kneels then stands with the
    /// help dial on (logged); the learned skill (M7) replaces it.</item>
    /// </list>
    /// Help is off for the whole fall: falls are purely physical.
    /// </summary>
    public sealed class FallController
    {
        public const float ReactionSeconds = 0.35f;
        public const float ReactionTone = 0.6f;
        public const float SettledTone = 0.15f;
        public const float NeckTone = 0.4f;
        public const float SettledSpeed = 0.05f;   // m/s, every link
        public const float SettledHold = 0.25f;    // s
        public const float RestBeforeGettingUp = 1.0f;
        public const float GetUpSeconds = 3.5f;
        /// <summary>Angular damping on every link while she settles and rests: rigid shapes would
        /// roll on forever, a soft body doesn't.</summary>
        public const float SettlingAngularDamping = 1.5f;
        /// <summary>A fall that hasn't come to rest by now goes straight to getting up.</summary>
        public const float LongestFall = 12f;
        /// <summary>Nothing about a fall lasts longer than this: she is standing again.</summary>
        public const float LongestFallAndGetUp = 22f;

        private readonly PhysicsBody _body;
        private readonly DesktopSurfaces _surfaces;
        private readonly StandInMotion _motion = new();
        private readonly CanonicalPose _pose = new();
        private readonly CanonicalPose _from = new();
        private readonly Collider[] _feet;
        private Vector2 _direction;
        private float _unsupported, _calm;

        public FallPhase Phase { get; private set; } = FallPhase.Supported;
        public float PhaseTime { get; private set; }
        public string Trigger { get; private set; } = "";
        public int Falls { get; private set; }
        /// <summary>When the first part other than a foot touched a surface (scenario time).</summary>
        public float? ImpactTime { get; private set; }
        public float? SettleSeconds { get; private set; }
        public float RestGap { get; private set; } = float.NaN;
        public float RestPenetration { get; private set; } = float.NaN;
        public bool AutoGetUp { get; set; } = true;
        /// <summary>While true, nothing starts a fall (she is being held, or something else is
        /// carrying her through the air).</summary>
        public bool Suspended { get; set; }
        /// <summary>While true (running), both feet leave the ground on purpose, so only a long
        /// time without support counts as losing it.</summary>
        public bool ExpectFlight { get; set; }
        /// <summary>The way she faces (simulation space): the fall's direction is judged from it,
        /// and she gets up facing it. The body's frame by default.</summary>
        public Quaternion Facing { get; set; }
        /// <summary>The first fall's results (the gate's): settling time, rest gap and penetration.</summary>
        public float? FirstSettleSeconds { get; private set; }
        public float FirstRestGap { get; private set; } = float.NaN;
        public float FirstRestPenetration { get; private set; } = float.NaN;
        /// <summary>What happened and when (scenario seconds).</summary>
        public System.Collections.Generic.List<string> Events { get; } = new();
        public bool Supported { get; private set; } = true;
        /// <summary>The help used outside falls (the scenario's dial).</summary>
        public HelpDial NormalHelp { get; set; }
        /// <summary>Help the stand-in get-up uses (logged like any other help).</summary>
        public HelpDial GetUpHelp { get; set; } = new(0, 1, 1);

        public FallController(PhysicsBody body, DesktopSurfaces surfaces)
        {
            _body = body ?? throw new ArgumentNullException(nameof(body));
            _surfaces = surfaces ?? throw new ArgumentNullException(nameof(surfaces));
            _feet = body.Colliders.Where(c => c.name.EndsWith("Foot") || c.name.EndsWith("Toes")).ToArray();
            NormalHelp = body.Help;
            Facing = body.Frame;
        }

        /// <summary>Starts lying where she is, as if she had just come to rest (the get-up scenario).</summary>
        public void StartResting(float time)
        {
            NormalHelp = _body.Help;
            _body.Help = HelpDial.None;
            Phase = FallPhase.Settling;
            ImpactTime = time;
            _fallTime = 0;
            _body.SetTone(j => Canonical.Names[j] is "neck" or "head" ? NeckTone : SettledTone);
            _body.SetAngularDamping(SettlingAngularDamping);
            Enter(FallPhase.Rested, time + SettledHold);
        }

        /// <summary>Being dropped (released mid-air).</summary>
        public void Drop() => Begin("dropped");

        /// <summary>A stumble: she starts falling from wherever she is (tests and diagnostics).</summary>
        public void Trip() => Begin("tripped");

        /// <summary>
        /// Advances one physics step. <paramref name="wanted"/> is what the motion source wants
        /// while she is supported; the returned pose is what the motors should follow.
        /// </summary>
        public CanonicalPose Step(float dt, float time, CanonicalPose wanted)
        {
            PhaseTime += dt;
            if (Phase != FallPhase.Supported)
            {
                _fallTime += dt;
                // Too long: get up from wherever she is (never snap upright), and in the end stand.
                if (_fallTime > LongestFall && Phase is FallPhase.Reacting or FallPhase.Settling or FallPhase.Rested)
                {
                    Events.Add($"{time:0.00} s fall timed out in {Phase}");
                    CaptureFrom();
                    Enter(FallPhase.GettingUp, time);
                }
                else if (_fallTime > LongestFallAndGetUp)
                {
                    Events.Add($"{time:0.00} s getting up timed out");
                    Enter(FallPhase.Supported, time);
                    return wanted;
                }
            }
            Supported = AnyFootDown(null);
            switch (Phase)
            {
                case FallPhase.Supported:
                    if (Suspended) { _unsupported = 0; return wanted; }
                    _unsupported = Supported ? 0 : _unsupported + dt;
                    var tilt = Vector3.Angle(_body.Links[0].transform.up, Vector3.up);
                    var falling = _body.Links[0].linearVelocity.y < -0.3f;
                    // No foot on anything for a moment: support is gone (a closing window, a drop).
                    if (ExpectFlight ? _unsupported > 0.4f : _unsupported > 0.15f || (_unsupported > 0.05f && falling)) Begin("lost support");
                    else if (tilt > 40f && !Supported) Begin("tipped over");
                    else if (tilt > 55f) Begin("tipped over");
                    if (Phase == FallPhase.Supported) return wanted;
                    break;
                case FallPhase.Reacting:
                    if (TouchingWithBody() && ImpactTime is null) ImpactTime = time;
                    if (PhaseTime >= ReactionSeconds || ImpactTime is not null) Enter(FallPhase.Settling, time);
                    break;
                case FallPhase.Settling:
                    if (ImpactTime is null && TouchingWithBody()) ImpactTime = time;
                    var fastest = MeasuredSpeed(dt);
                    _calm = fastest < SettledSpeed ? _calm + dt : 0;
                    if (_calm == 0 && fastest < float.MaxValue)
                        LastMover = $"{_body.Links[_fastestLink].name} {fastest:0.00} m/s at {time:0.00} s";
                    if (_calm >= SettledHold && ImpactTime is not null) Enter(FallPhase.Rested, time);
                    // Still, and nothing but her feet ever touched: she caught herself.
                    else if (_calm >= SettledHold && Supported)
                    {
                        Events.Add($"{time:0.00} s caught herself");
                        Enter(FallPhase.Supported, time);
                    }
                    break;
                case FallPhase.Rested:
                    if (AutoGetUp && PhaseTime >= RestBeforeGettingUp) Enter(FallPhase.GettingUp, time);
                    break;
                case FallPhase.GettingUp:
                    // Done once both feet have been on the ground for a moment (or after 3 s more).
                    _planted = BothFeetDown() ? _planted + dt : 0;
                    if (PhaseTime >= GetUpSeconds && (_planted >= 0.3f || PhaseTime >= GetUpSeconds + 3f)) Enter(FallPhase.Supported, time);
                    break;
            }
            return PoseFor(wanted);
        }

        private CanonicalPose PoseFor(CanonicalPose wanted)
        {
            switch (Phase)
            {
                case FallPhase.Reacting:
                    _motion.FallReaction(_direction, _pose);
                    return _pose;
                case FallPhase.Settling:
                case FallPhase.Rested:
                    _motion.Relaxed(_pose);
                    return _pose;
                case FallPhase.GettingUp:
                {
                    // Stand-in: from the settled pose to kneeling, then to the wanted pose.
                    var t = Mathf.Min(PhaseTime / GetUpSeconds, 1f);
                    var kneel = new CanonicalPose();
                    _motion.Kneel(kneel);
                    var (a, b, s) = t < 0.45f ? (_from, kneel, t / 0.45f) : (kneel, wanted, (t - 0.45f) / 0.55f);
                    s = Mathf.SmoothStep(0, 1, s);
                    for (var j = 0; j < Canonical.Count; j++) _pose.Local[j] = Quaternion.Slerp(a.Local[j], b.Local[j], s);
                    _pose.Root = Vector3.Lerp(a.Root, b.Root, s);
                    // The balance helper lifts her hips from where they lie towards standing height.
                    var hips = _body.Links[0].transform.position;
                    var standing = _body.SimPoint(_body.Fit.HipsFor(wanted.Root));
                    _body.BalanceTargetPosition = new Vector3(hips.x, Mathf.Lerp(hips.y, standing.y, s * s), hips.z);
                    _body.BalanceTargetRotation = Quaternion.Slerp(_body.Links[0].transform.rotation, Facing, Mathf.Clamp01(t * 2));
                    return _pose;
                }
                default:
                    return wanted;
            }
        }

        private void Begin(string trigger)
        {
            if (Phase != FallPhase.Supported) return;
            Trigger = trigger;
            Falls++;
            Events.Add($"{PhaseTime:0.00} s into {Phase}: fall ({trigger})");
            ImpactTime = null;
            SettleSeconds = null;
            RestGap = RestPenetration = float.NaN;
            var v = _body.Links[0].linearVelocity;
            var local = Quaternion.Inverse(Facing) * new Vector3(v.x, 0, v.z);
            var tiltAxis = Quaternion.Inverse(Facing) * _body.Links[0].transform.up;
            _direction = local.sqrMagnitude > 0.01f ? new Vector2(local.x, local.z) : new Vector2(tiltAxis.x, tiltAxis.z);
            NormalHelp = _body.Help;
            _body.Help = HelpDial.None;
            _body.SetTone(_ => ReactionTone);
            Phase = FallPhase.Reacting;
            PhaseTime = 0;
            _fallTime = 0;
            _historyNext = _historyCount = 0;
        }

        private void Enter(FallPhase phase, float time)
        {
            Events.Add($"{time:0.00} s {phase}");
            Phase = phase;
            PhaseTime = 0;
            _calm = 0;
            switch (phase)
            {
                case FallPhase.Settling:
                    _body.SetTone(j => Canonical.Names[j] is "neck" or "head" ? NeckTone : SettledTone);
                    _body.SetAngularDamping(SettlingAngularDamping);
                    break;
                case FallPhase.Rested:
                    // When she became still (the confirmation hold isn't part of settling).
                    SettleSeconds = time - SettledHold - ImpactTime!.Value;
                    MeasureRest();
                    if (FirstSettleSeconds is null)
                    {
                        FirstSettleSeconds = SettleSeconds;
                        FirstRestGap = RestGap;
                        FirstRestPenetration = RestPenetration;
                    }
                    CaptureFrom();
                    break;
                case FallPhase.GettingUp:
                    _body.Help = GetUpHelp;
                    _body.SetTone(_ => 1f);
                    _body.SetAngularDamping(null);
                    break;
                case FallPhase.Supported:
                    _body.Help = NormalHelp;
                    _body.SetTone(_ => 1f);
                    _body.SetAngularDamping(null);
                    _unsupported = 0;
                    break;
            }
        }

        /// <summary>The pose getting up starts from: where she lies now.</summary>
        private void CaptureFrom()
        {
            for (var j = 0; j < Canonical.Count; j++) _from.Local[j] = Quaternion.identity;
            var realised = _body.ReadNormalized();
            for (var i = 0; i < realised.Length; i++) _from.Local[_body.Fit.Map.Joints[i]] = realised[i];
            _from.Root = new Vector3(0, 0.2f, 0);
        }

        /// <summary>How far back the speed check looks: long enough that contact chatter (a toe
        /// buzzing a millimetre per step against the floor, too small to see) averages out, short
        /// enough that real sliding or rolling still counts.</summary>
        public const float SpeedWindow = 0.1f; // s

        private Vector3[][] _history = Array.Empty<Vector3[]>();
        private float[] _historyTime = Array.Empty<float>();
        private int _historyNext, _historyCount;
        private float _historyClock;
        private int _fastestLink;

        /// <summary>The last link that kept her from settling, while she settled (diagnostics).</summary>
        public string? LastMover { get; private set; }

        /// <summary>The fastest link's average speed over the last <see cref="SpeedWindow"/>, from
        /// positions. (Non-root links' reported velocities can carry a steady offset at rest, and
        /// one step's movement is dominated by contact chatter, so positions over a window decide.)</summary>
        private float MeasuredSpeed(float dt)
        {
            var links = _body.Links;
            var slots = Mathf.CeilToInt(SpeedWindow / Mathf.Max(dt, 1e-4f)) + 1;
            if (_history.Length != slots || _history[0].Length != links.Count)
            {
                _history = new Vector3[slots][];
                for (var s = 0; s < slots; s++) _history[s] = new Vector3[links.Count];
                _historyTime = new float[slots];
                _historyNext = _historyCount = 0;
            }
            _historyClock += dt;
            var now = _history[_historyNext];
            for (var i = 0; i < links.Count; i++) now[i] = links[i].transform.position;
            _historyTime[_historyNext] = _historyClock;
            var oldest = _historyCount < slots ? 0 : (_historyNext + 1) % slots;
            _historyNext = (_historyNext + 1) % slots;
            _historyCount = Mathf.Min(_historyCount + 1, slots);
            var span = _historyClock - _historyTime[oldest];
            if (_historyCount < 2 || span <= 0) return float.MaxValue; // not measured yet: not calm
            var then = _history[oldest];
            var fastest = 0f;
            for (var i = 0; i < links.Count; i++)
            {
                var speed = Vector3.Distance(now[i], then[i]) / span;
                if (speed > fastest) { fastest = speed; _fastestLink = i; }
            }
            return fastest;
        }

        private float _planted, _fallTime;

        // These run every physics step, so they avoid LINQ and allocations.
        private bool AnyFootDown(string? side)
        {
            foreach (var f in _feet)
                if ((side == null || f.name.Contains(side)) && _surfaces.Contact(f).Gap < 0.01f) return true;
            return false;
        }

        private bool BothFeetDown() => AnyFootDown("left") && AnyFootDown("right");

        private bool TouchingWithBody()
        {
            foreach (var c in _body.Colliders)
                if (Array.IndexOf(_feet, c) < 0 && _surfaces.Contact(c).Gap < 0.005f) return true;
            return false;
        }

        /// <summary>The fall gate's measurement: the closest body part's gap to a surface and the
        /// deepest penetration, once she has come to rest.</summary>
        public void MeasureRest()
        {
            var gap = float.MaxValue; var penetration = 0f;
            foreach (var c in _body.Colliders)
            {
                var contact = _surfaces.Contact(c);
                gap = Mathf.Min(gap, contact.Gap);
                penetration = Mathf.Max(penetration, contact.Penetration);
            }
            RestGap = gap;
            RestPenetration = penetration;
        }
    }
}
