using System;
using System.IO;

namespace MotionSpike
{
    /// <summary>
    /// motion-lab's StepModel, one tick, in plain C# (no packages): GELU encoders for the motion
    /// state and the MotionSignal features, a GRU (one or two layers, PyTorch's gate order r, z, n)
    /// and a linear decoder to 6D rotations. Weights come from scripts/spike_export.py's
    /// .weights.bin, in PyTorch's state_dict order.
    /// </summary>
    public sealed class CsGru
    {
        public const int PoseWidth = 256, SignalWidth = 64;
        public readonly int StateSize, SignalSize, OutSize, Hidden, Layers;
        private readonly float[] _wPose, _bPose, _wSignal, _bSignal, _wOut, _bOut;
        private readonly float[][] _wIh, _wHh, _bIh, _bHh;
        private readonly float[] _x, _gi, _gh, _pose, _signal;

        public CsGru(string weightsPath, int stateSize, int signalSize, int outSize, int hidden, int layers)
        {
            StateSize = stateSize; SignalSize = signalSize; OutSize = outSize; Hidden = hidden; Layers = layers;
            var bytes = File.ReadAllBytes(weightsPath);
            var all = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, all, 0, bytes.Length);
            var at = 0;
            float[] Take(int count)
            {
                var part = new float[count];
                Array.Copy(all, at, part, 0, count);
                at += count;
                return part;
            }
            _wPose = Take(PoseWidth * stateSize); _bPose = Take(PoseWidth);
            _wSignal = Take(SignalWidth * signalSize); _bSignal = Take(SignalWidth);
            _wIh = new float[layers][]; _wHh = new float[layers][]; _bIh = new float[layers][]; _bHh = new float[layers][];
            for (var l = 0; l < layers; l++)
            {
                var input = l == 0 ? PoseWidth + SignalWidth : hidden;
                _wIh[l] = Take(3 * hidden * input); _wHh[l] = Take(3 * hidden * hidden);
                _bIh[l] = Take(3 * hidden); _bHh[l] = Take(3 * hidden);
            }
            _wOut = Take(outSize * hidden); _bOut = Take(outSize);
            if (at != all.Length) throw new InvalidDataException($"weights: used {at} of {all.Length} floats");
            _x = new float[Math.Max(PoseWidth + SignalWidth, hidden)];
            _gi = new float[3 * hidden]; _gh = new float[3 * hidden];
            _pose = new float[PoseWidth]; _signal = new float[SignalWidth];
        }

        public int Parameters =>
            _wPose.Length + _bPose.Length + _wSignal.Length + _bSignal.Length + _wOut.Length + _bOut.Length +
            Sum(_wIh) + Sum(_wHh) + Sum(_bIh) + Sum(_bHh);

        private static int Sum(float[][] parts) { var n = 0; foreach (var p in parts) n += p.Length; return n; }

        /// <summary>One tick. <paramref name="h"/> is (layers × hidden) and is updated in place.</summary>
        public void Step(float[] state, float[] signal, float[] h, float[] rot6d)
        {
            Linear(_wPose, _bPose, state, StateSize, _pose, PoseWidth);
            for (var i = 0; i < PoseWidth; i++) _x[i] = Gelu(_pose[i]);
            Linear(_wSignal, _bSignal, signal, SignalSize, _signal, SignalWidth);
            for (var i = 0; i < SignalWidth; i++) _x[PoseWidth + i] = Gelu(_signal[i]);
            var inputSize = PoseWidth + SignalWidth;
            for (var l = 0; l < Layers; l++)
            {
                var offset = l * Hidden;
                Linear(_wIh[l], _bIh[l], _x, inputSize, _gi, 3 * Hidden);
                LinearSlice(_wHh[l], _bHh[l], h, offset, Hidden, _gh, 3 * Hidden);
                for (var j = 0; j < Hidden; j++)
                {
                    var r = Sigmoid(_gi[j] + _gh[j]);
                    var z = Sigmoid(_gi[Hidden + j] + _gh[Hidden + j]);
                    var n = (float)Math.Tanh(_gi[2 * Hidden + j] + r * _gh[2 * Hidden + j]);
                    h[offset + j] = (1 - z) * n + z * h[offset + j];
                }
                Array.Copy(h, offset, _x, 0, Hidden);
                inputSize = Hidden;
            }
            Linear(_wOut, _bOut, _x, Hidden, rot6d, OutSize);
        }

        private static void Linear(float[] w, float[] b, float[] x, int inSize, float[] y, int outSize)
        {
            for (var o = 0; o < outSize; o++)
            {
                var sum = b[o];
                var row = o * inSize;
                for (var i = 0; i < inSize; i++) sum += w[row + i] * x[i];
                y[o] = sum;
            }
        }

        private static void LinearSlice(float[] w, float[] b, float[] x, int xOffset, int inSize, float[] y, int outSize)
        {
            for (var o = 0; o < outSize; o++)
            {
                var sum = b[o];
                var row = o * inSize;
                for (var i = 0; i < inSize; i++) sum += w[row + i] * x[xOffset + i];
                y[o] = sum;
            }
        }

        private static float Sigmoid(float v) => 1f / (1f + (float)Math.Exp(-v));

        /// <summary>PyTorch's exact GELU, 0.5·x·(1 + erf(x/√2)).</summary>
        private static float Gelu(float v) => (float)(0.5 * v * (1.0 + Erf(v / Math.Sqrt(2.0))));

        /// <summary>erf to about 1.2e-7 relative error (Numerical Recipes' erfc Chebyshev fit).</summary>
        public static double Erf(double x)
        {
            var z = Math.Abs(x);
            var t = 1.0 / (1.0 + 0.5 * z);
            var erfc = t * Math.Exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
                t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
                t * (-0.82215223 + t * 0.17087277)))))))));
            return x >= 0 ? 1.0 - erfc : erfc - 1.0;
        }
    }
}
