using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using MotionSpike;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;

/// <summary>
/// Roadmap M0 runtime spike: per-tick cost and parity of motion-lab's StepModel in Unity, for
/// every size of M1's sizing sweep, on three paths: Sentis (Unity Inference Engine 2.6) on the
/// CPU, Sentis on the GPU, and a hand-written C# GRU. Each path is checked against PyTorch's
/// 120-tick reference (hidden state fed back), then timed per tick including upload and readback.
///   Unity -batchmode -projectPath unity-spike -executeMethod SpikeRunner.Run -quit
/// Results: ../outputs/spike/unity_results.json
/// </summary>
public static class SpikeRunner
{
    private const int Warmup = 100, Timed = 1000;
    private static readonly string[] Names =
        { "step_128x1", "step_256x1", "step_512x1", "step_128x2", "step_256x2", "step_512x2" };

    public static void Run()
    {
        var spike = Path.Combine(Application.streamingAssetsPath, "spike");
        var meta = File.ReadAllText(Path.Combine(spike, "models.json"));
        int Read(string key) => int.Parse(System.Text.RegularExpressions.Regex.Match(meta, $"\"{key}\": (\\d+)").Groups[1].Value);
        int stateSize = Read("state_size"), signalSize = Read("signal_size"), outSize = Read("rot6d_size"), ticks = Read("ticks");

        var json = new StringBuilder();
        json.Append("{\n \"unity\": \"").Append(Application.unityVersion)
            .Append("\",\n \"graphics_device\": \"").Append(SystemInfo.graphicsDeviceName)
            .Append("\",\n \"graphics_api\": \"").Append(SystemInfo.graphicsDeviceType)
            .Append("\",\n \"cpu\": \"").Append(SystemInfo.processorType)
            .Append("\",\n \"scripting\": \"editor (Mono)\",\n \"warmup_ticks\": ").Append(Warmup)
            .Append(",\n \"timed_ticks\": ").Append(Timed).Append(",\n \"results\": [\n");
        var first = true;
        foreach (var name in Names)
        {
            var hidden = int.Parse(name.Split('_')[1].Split('x')[0]);
            var layers = int.Parse(name.Split('x')[1]);
            var reference = LoadReference(Path.Combine(spike, name + ".ref.bin"), ticks, stateSize, signalSize, outSize, layers * hidden);
            foreach (var path in new[] { "sentis_cpu", "sentis_gpu", "csharp_gru" })
            {
                string row;
                try
                {
                    row = path == "csharp_gru"
                        ? MeasureCSharp(name, spike, reference, stateSize, signalSize, outSize, hidden, layers)
                        : MeasureSentis(name, path == "sentis_cpu" ? BackendType.CPU : BackendType.GPUCompute,
                            reference, stateSize, signalSize, outSize, hidden, layers);
                }
                catch (Exception error)
                {
                    row = $"{{\"model\": \"{name}\", \"path\": \"{path}\", \"error\": \"{Escape(error.GetType().Name + ": " + error.Message)}\"}}";
                }
                Debug.Log("SPIKE " + row);
                json.Append(first ? "  " : ",\n  ").Append(row);
                first = false;
            }
        }
        json.Append("\n ]\n}\n");
        var output = Path.GetFullPath(Path.Combine(Application.dataPath, "../../outputs/spike/unity_results.json"));
        File.WriteAllText(output, json.ToString());
        Debug.Log("SPIKE wrote " + output);
    }

    private sealed class Reference
    {
        public float[][] State, Signal, Rot6d, H;
    }

    private static Reference LoadReference(string path, int ticks, int stateSize, int signalSize, int outSize, int hSize)
    {
        var bytes = File.ReadAllBytes(path);
        var all = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, all, 0, bytes.Length);
        var row = stateSize + signalSize + outSize + hSize;
        if (all.Length != ticks * row) throw new InvalidDataException($"{path}: {all.Length} floats, expected {ticks * row}");
        var reference = new Reference { State = new float[ticks][], Signal = new float[ticks][], Rot6d = new float[ticks][], H = new float[ticks][] };
        for (var t = 0; t < ticks; t++)
        {
            var at = t * row;
            reference.State[t] = Slice(all, at, stateSize); at += stateSize;
            reference.Signal[t] = Slice(all, at, signalSize); at += signalSize;
            reference.Rot6d[t] = Slice(all, at, outSize); at += outSize;
            reference.H[t] = Slice(all, at, hSize);
        }
        return reference;
    }

    private static float[] Slice(float[] all, int at, int count)
    {
        var part = new float[count];
        Array.Copy(all, at, part, 0, count);
        return part;
    }

    private static string MeasureSentis(string name, BackendType backend, Reference reference,
        int stateSize, int signalSize, int outSize, int hidden, int layers)
    {
        var load = Stopwatch.StartNew();
        var asset = AssetDatabase.LoadAssetAtPath<ModelAsset>($"Assets/Models/{name}.onnx");
        if (asset == null) throw new FileNotFoundException("ModelAsset not imported: " + name);
        var model = ModelLoader.Load(asset);
        using var worker = new Worker(model, backend);
        load.Stop();
        using var state = new Tensor<float>(new TensorShape(1, stateSize));
        using var signal = new Tensor<float>(new TensorShape(1, signalSize));
        using var h = new Tensor<float>(new TensorShape(layers, 1, hidden));
        var hValues = new float[layers * hidden];

        float[] rot = null;
        void Tick(float[] s, float[] g)
        {
            state.Upload(s);
            signal.Upload(g);
            h.Upload(hValues);
            worker.Schedule(state, signal, h);
            rot = ((Tensor<float>)worker.PeekOutput("rot6d")).DownloadToArray();
            hValues = ((Tensor<float>)worker.PeekOutput("h_next")).DownloadToArray();
        }

        var worst = 0f;
        for (var t = 0; t < reference.State.Length; t++)
        {
            Tick(reference.State[t], reference.Signal[t]);
            worst = Math.Max(worst, MaxDiff(rot, reference.Rot6d[t]));
            worst = Math.Max(worst, MaxDiff(hValues, reference.H[t]));
        }
        var timings = Time(reference, (s, g) => Tick(s, g), () => Array.Clear(hValues, 0, hValues.Length));
        var finite = rot.All(v => !float.IsNaN(v) && !float.IsInfinity(v));
        return Row(name, backend == BackendType.CPU ? "sentis_cpu" : "sentis_gpu", worst, timings, load.Elapsed.TotalMilliseconds, finite);
    }

    private static string MeasureCSharp(string name, string spike, Reference reference,
        int stateSize, int signalSize, int outSize, int hidden, int layers)
    {
        var load = Stopwatch.StartNew();
        var gru = new CsGru(Path.Combine(spike, name + ".weights.bin"), stateSize, signalSize, outSize, hidden, layers);
        load.Stop();
        var h = new float[layers * hidden];
        var rot = new float[outSize];
        var worst = 0f;
        for (var t = 0; t < reference.State.Length; t++)
        {
            gru.Step(reference.State[t], reference.Signal[t], h, rot);
            worst = Math.Max(worst, MaxDiff(rot, reference.Rot6d[t]));
            worst = Math.Max(worst, MaxDiff(h, reference.H[t]));
        }
        var timings = Time(reference, (s, g) => gru.Step(s, g, h, rot), () => Array.Clear(h, 0, h.Length));
        var finite = rot.All(v => !float.IsNaN(v) && !float.IsInfinity(v));
        return Row(name, "csharp_gru", worst, timings, load.Elapsed.TotalMilliseconds, finite);
    }

    private static double[] Time(Reference reference, Action<float[], float[]> tick, Action reset)
    {
        reset();
        var n = reference.State.Length;
        for (var t = 0; t < Warmup; t++) tick(reference.State[t % n], reference.Signal[t % n]);
        var micro = new double[Timed];
        var clock = new Stopwatch();
        for (var t = 0; t < Timed; t++)
        {
            clock.Restart();
            tick(reference.State[t % n], reference.Signal[t % n]);
            clock.Stop();
            micro[t] = clock.Elapsed.TotalMilliseconds * 1000.0;
        }
        Array.Sort(micro);
        return micro;
    }

    private static string Row(string name, string path, float parity, double[] sortedMicro, double loadMs, bool finite)
    {
        string F(double v) => v.ToString("0.0", CultureInfo.InvariantCulture);
        double P(double q) => sortedMicro[Math.Min(sortedMicro.Length - 1, (int)(q * sortedMicro.Length))];
        return $"{{\"model\": \"{name}\", \"path\": \"{path}\", \"parity_max_abs\": {parity.ToString("E2", CultureInfo.InvariantCulture)}, " +
               $"\"finite\": {(finite ? "true" : "false")}, \"load_ms\": {F(loadMs)}, \"tick_us\": {{\"mean\": {F(sortedMicro.Average())}, " +
               $"\"p50\": {F(P(0.5))}, \"p99\": {F(P(0.99))}, \"max\": {F(sortedMicro[^1])}}}}}";
    }

    private static float MaxDiff(float[] a, float[] b)
    {
        if (a.Length != b.Length) throw new InvalidDataException($"length {a.Length} vs {b.Length}");
        var worst = 0f;
        for (var i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        return worst;
    }

    private static string Escape(string text) => text.Replace("\\", "\\\\").Replace("\"", "'").Replace("\n", " ");
}
