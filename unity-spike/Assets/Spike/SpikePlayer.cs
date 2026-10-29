using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Unity.InferenceEngine;
using UniVRM10;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace MotionSpike
{
    /// <summary>
    /// Roadmap M0 runtime spike in a built player: Alice (or any VRM) driven at 60 Hz by
    /// motion-lab's StepModel, one tick per frame. Each tick builds the motion state from her
    /// realised pose (6D rotations, ω by finite difference, zero root velocity), runs the model,
    /// turns its 6D output into rotations, folds missing joints (upperChest into chest), converts
    /// to Unity's frame and writes her control-rig bones. The whole tick is timed, and the model
    /// call on its own.
    ///   Spike.exe -vrm Alice.vrm -out results.json [-seconds 10] [-shots folder]
    /// The spike's weights are random, so each joint's rotation is scaled to 15 % of its angle
    /// for display: the cost is what is measured, not the motion.
    /// </summary>
    public sealed class SpikePlayer : MonoBehaviour
    {
        public ModelAsset[] models = Array.Empty<ModelAsset>();
        public string[] modelNames = Array.Empty<string>();

        private static readonly HumanBodyBones[] Canonical =
        {
            HumanBodyBones.Hips, HumanBodyBones.Spine, HumanBodyBones.Chest, HumanBodyBones.UpperChest,
            HumanBodyBones.Neck, HumanBodyBones.Head,
            HumanBodyBones.LeftShoulder, HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm, HumanBodyBones.LeftHand,
            HumanBodyBones.RightShoulder, HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm, HumanBodyBones.RightHand,
            HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftFoot, HumanBodyBones.LeftToes,
            HumanBodyBones.RightUpperLeg, HumanBodyBones.RightLowerLeg, HumanBodyBones.RightFoot, HumanBodyBones.RightToes,
        };
        private const int J = 22, StateSize = J * 6 + J * 3 + 3, SignalSize = 26, OutSize = J * 6;
        private const float DisplayScale = 0.15f, Dt = 1f / 60f;

        private readonly Transform[] _bones = new Transform[J];
        private readonly Quaternion[] _realised = new Quaternion[J];
        private readonly Quaternion[] _previous = new Quaternion[J];
        private readonly float[] _state = new float[StateSize];
        private readonly float[] _signal = new float[SignalSize];
        private readonly float[] _rot6d = new float[OutSize];
        private string _out = "spike_results.json", _shots = "", _vrm = "";
        private float _seconds = 10f;
        private readonly StringBuilder _json = new StringBuilder();
        private string _error = "";

        private interface IStep : IDisposable
        {
            void Reset();
            void Step(float[] state, float[] signal, float[] rot6d);
        }

        private sealed class SentisStep : IStep
        {
            private readonly Worker _worker;
            private readonly Tensor<float> _state, _signal, _h;
            private float[] _hValues;

            public SentisStep(ModelAsset asset, BackendType backend, int hidden, int layers)
            {
                _worker = new Worker(ModelLoader.Load(asset), backend);
                _state = new Tensor<float>(new TensorShape(1, StateSize));
                _signal = new Tensor<float>(new TensorShape(1, SignalSize));
                _h = new Tensor<float>(new TensorShape(layers, 1, hidden));
                _hValues = new float[layers * hidden];
            }

            public void Reset() => Array.Clear(_hValues, 0, _hValues.Length);

            public void Step(float[] state, float[] signal, float[] rot6d)
            {
                _state.Upload(state);
                _signal.Upload(signal);
                _h.Upload(_hValues);
                _worker.Schedule(_state, _signal, _h);
                var output = ((Tensor<float>)_worker.PeekOutput("rot6d")).DownloadToArray();
                Array.Copy(output, rot6d, OutSize);
                _hValues = ((Tensor<float>)_worker.PeekOutput("h_next")).DownloadToArray();
            }

            public void Dispose()
            {
                _worker.Dispose(); _state.Dispose(); _signal.Dispose(); _h.Dispose();
            }
        }

        private sealed class CSharpStep : IStep
        {
            private readonly CsGru _gru;
            private readonly float[] _h;

            public CSharpStep(string weights, int hidden, int layers)
            {
                _gru = new CsGru(weights, StateSize, SignalSize, OutSize, hidden, layers);
                _h = new float[layers * hidden];
            }

            public void Reset() => Array.Clear(_h, 0, _h.Length);
            public void Step(float[] state, float[] signal, float[] rot6d) => _gru.Step(state, signal, _h, rot6d);
            public void Dispose() { }
        }

        private async void Start()
        {
            Application.targetFrameRate = 60;
            QualitySettings.vSyncCount = 0;
            var args = Environment.GetCommandLineArgs();
            for (var i = 0; i < args.Length - 1; i++)
            {
                if (args[i] == "-vrm") _vrm = args[i + 1];
                if (args[i] == "-out") _out = args[i + 1];
                if (args[i] == "-shots") _shots = args[i + 1];
                if (args[i] == "-seconds") _seconds = float.Parse(args[i + 1], CultureInfo.InvariantCulture);
            }
            try
            {
                var instance = await Vrm10.LoadPathAsync(_vrm, canLoadVrm0X: true,
                    controlRigGenerationOption: ControlRigGenerationOption.Generate);
                var rig = instance.Runtime.ControlRig;
                for (var j = 0; j < J; j++) _bones[j] = rig.GetBoneTransform(Canonical[j]);
                var head = instance.GetComponent<Animator>().GetBoneTransform(HumanBodyBones.Head).position;
                var cam = Camera.main!;
                cam.transform.position = new Vector3(0, head.y * 0.55f, 2.6f);
                cam.transform.LookAt(new Vector3(0, head.y * 0.55f, 0));
                StartCoroutine(RunAll());
            }
            catch (Exception error)
            {
                _error = error.GetType().Name + ": " + error.Message;
                Finish();
            }
        }

        private IEnumerator RunAll()
        {
            var weightsFolder = Path.Combine(Application.streamingAssetsPath, "spike");
            var first = true;
            for (var m = 0; m < models.Length; m++)
            {
                var name = modelNames[m];
                var hidden = int.Parse(name.Split('_')[1].Split('x')[0]);
                var layers = int.Parse(name.Split('x')[1]);
                foreach (var path in new[] { "sentis_cpu", "sentis_gpu", "csharp_gru" })
                {
                    IStep step;
                    var load = Stopwatch.StartNew();
                    try
                    {
                        step = path == "csharp_gru"
                            ? new CSharpStep(Path.Combine(weightsFolder, name + ".weights.bin"), hidden, layers)
                            : new SentisStep(models[m], path == "sentis_cpu" ? BackendType.CPU : BackendType.GPUCompute, hidden, layers);
                    }
                    catch (Exception error)
                    {
                        Append(ref first, $"{{\"model\": \"{name}\", \"path\": \"{path}\", \"error\": \"{error.Message.Replace("\"", "'")}\"}}");
                        continue;
                    }
                    load.Stop();
                    using (step)
                    {
                        step.Reset();
                        for (var j = 0; j < J; j++) { _realised[j] = Quaternion.identity; _previous[j] = Quaternion.identity; }
                        var frames = Mathf.RoundToInt(_seconds * 60);
                        var tick = new List<double>(frames);
                        var model = new List<double>(frames);
                        var frame = new List<double>(frames);
                        var finite = true;
                        var clock = new Stopwatch();
                        var modelClock = new Stopwatch();
                        for (var f = 0; f < 60 + frames; f++)
                        {
                            clock.Restart();
                            BuildState(f);
                            modelClock.Restart();
                            step.Step(_state, _signal, _rot6d);
                            modelClock.Stop();
                            finite &= Apply();
                            clock.Stop();
                            if (f >= 60)
                            {
                                tick.Add(clock.Elapsed.TotalMilliseconds * 1000);
                                model.Add(modelClock.Elapsed.TotalMilliseconds * 1000);
                                frame.Add(Time.unscaledDeltaTime * 1000);
                            }
                            if (f == 60 + frames / 2 && _shots.Length > 0 && name == "step_256x1")
                                ScreenCapture.CaptureScreenshot(Path.Combine(_shots, $"{name}_{path}.png"));
                            yield return null;
                        }
                        Append(ref first, Row(name, path, load.Elapsed.TotalMilliseconds, tick, model, frame, finite));
                    }
                }
            }
            Finish();
        }

        private void BuildState(int frame)
        {
            var at = 0;
            for (var j = 0; j < J; j++)
            {
                var m = Matrix4x4.Rotate(_realised[j]);
                _state[at++] = m.m00; _state[at++] = m.m10; _state[at++] = m.m20;
                _state[at++] = m.m01; _state[at++] = m.m11; _state[at++] = m.m21;
            }
            for (var j = 0; j < J; j++)
            {
                // Body-frame ω = Log(R_prevᵀ·R)/Δt, from the realised pose.
                (Quaternion.Inverse(_previous[j]) * _realised[j]).ToAngleAxis(out var angle, out var axis);
                if (angle > 180f) angle -= 360f;
                var w = float.IsNaN(axis.x) ? Vector3.zero : axis * (angle * Mathf.Deg2Rad / Dt);
                _state[at++] = w.x; _state[at++] = w.y; _state[at++] = w.z;
            }
            _state[at++] = 0; _state[at++] = 0; _state[at] = 0;
            for (var i = 0; i < SignalSize; i++) _signal[i] = 0;
            _signal[0] = 1;  // a neutral intent one-hot
            _signal[20] = 0.5f + 0.5f * Mathf.Sin(frame * 0.05f);  // a slowly varying feature
        }

        /// <summary>6D → rotation, display scale, fold missing joints, mirror into Unity, apply.</summary>
        private bool Apply()
        {
            var finite = true;
            for (var j = 0; j < J; j++)
            {
                var o = j * 6;
                var a1 = new Vector3(_rot6d[o], _rot6d[o + 1], _rot6d[o + 2]);
                var a2 = new Vector3(_rot6d[o + 3], _rot6d[o + 4], _rot6d[o + 5]);
                var b1 = a1.normalized;
                var b2 = (a2 - Vector3.Dot(b1, a2) * b1).normalized;
                var b3 = Vector3.Cross(b1, b2);
                finite &= !float.IsNaN(b3.x + b3.y + b3.z);
                var m = Matrix4x4.identity;
                m.SetColumn(0, b1); m.SetColumn(1, b2); m.SetColumn(2, b3);
                var q = Quaternion.Slerp(Quaternion.identity, m.rotation, DisplayScale);
                _previous[j] = _realised[j];
                _realised[j] = finite ? q : Quaternion.identity;
            }
            for (var j = 0; j < J; j++)
            {
                var q = _realised[j];
                if (j == 2 && _bones[3] == null) q *= _realised[3];  // chest ← chest · upperChest
                if (_bones[j] == null) continue;
                _bones[j].localRotation = new Quaternion(q.x, -q.y, -q.z, q.w);  // motion-lab (right-handed) → Unity
            }
            return finite;
        }

        private void Append(ref bool first, string row)
        {
            Debug.Log("SPIKE " + row);
            _json.Append(first ? "  " : ",\n  ").Append(row);
            first = false;
        }

        private static string Row(string name, string path, double loadMs, List<double> tick, List<double> model,
            List<double> frame, bool finite)
        {
            string F(double v) => v.ToString("0.0", CultureInfo.InvariantCulture);
            string Stats(List<double> values)
            {
                var sorted = values.OrderBy(v => v).ToArray();
                double P(double q) => sorted[Math.Min(sorted.Length - 1, (int)(q * sorted.Length))];
                return $"{{\"mean\": {F(sorted.Average())}, \"p50\": {F(P(0.5))}, \"p99\": {F(P(0.99))}, \"max\": {F(sorted[^1])}}}";
            }
            return $"{{\"model\": \"{name}\", \"path\": \"{path}\", \"load_ms\": {F(loadMs)}, \"finite\": {(finite ? "true" : "false")}, " +
                   $"\"ticks\": {tick.Count}, \"tick_us\": {Stats(tick)}, \"model_us\": {Stats(model)}, \"frame_ms\": {Stats(frame)}}}";
        }

        private void Finish()
        {
            var text = new StringBuilder();
            text.Append("{\n \"unity\": \"").Append(Application.unityVersion)
                .Append("\",\n \"development_build\": ").Append(Debug.isDebugBuild ? "true" : "false")
                .Append(",\n \"graphics_device\": \"").Append(SystemInfo.graphicsDeviceName)
                .Append("\",\n \"cpu\": \"").Append(SystemInfo.processorType)
                .Append("\",\n \"vrm\": \"").Append(Path.GetFileName(_vrm))
                .Append("\",\n \"error\": \"").Append(_error.Replace("\"", "'"))
                .Append("\",\n \"results\": [\n").Append(_json).Append("\n ]\n}\n");
            File.WriteAllText(_out, text.ToString());
            Application.Quit();
        }
    }
}
