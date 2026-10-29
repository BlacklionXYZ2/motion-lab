using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace MotionEnv.AvatarLayer
{
    /// <summary>
    /// Desktop surfaces as solid colliders: the taskbar, window tops and screen edges. The body
    /// simulates in metres at the avatar's own size, so pixels are converted by her height:
    /// in SORA she stands <see cref="SoraHeightPixels"/> physical pixels tall.
    /// </summary>
    public sealed class DesktopSurfaces : IDisposable
    {
        /// <summary>Her height on screen in SORA (owner, 2026-10-16).</summary>
        public const float SoraHeightPixels = 500f;

        public readonly struct Surface
        {
            public readonly string Name;
            public readonly Vector3 Center, Size;

            public Surface(string name, Vector3 center, Vector3 size)
            {
                Name = name; Center = center; Size = size;
            }

            /// <summary>A screen rectangle's top edge as a ledge, from pixels (y grows downward,
            /// measured from the floor's top), for a body of the given height in metres.</summary>
            public static Surface FromPixels(string name, float left, float top, float width, float floorTop,
                float bodyHeightMetres, float depthMetres = 1f, float thicknessPixels = 30f)
            {
                var m = bodyHeightMetres / SoraHeightPixels;
                var center = new Vector3((left + width / 2) * m, (floorTop - top) * m - thicknessPixels * m / 2, 0);
                return new Surface(name, center, new Vector3(width * m, thicknessPixels * m, depthMetres));
            }
        }

        private readonly List<(Surface Surface, BoxCollider Collider)> _surfaces = new();

        public GameObject Container { get; }
        public IEnumerable<BoxCollider> Colliders => _surfaces.Select(s => s.Collider);

        private DesktopSurfaces()
        {
            Container = new GameObject("Desktop surfaces");
        }

        public static DesktopSurfaces Create(params Surface[] surfaces)
        {
            var desktop = new DesktopSurfaces();
            foreach (var s in surfaces) desktop.Add(s);
            return desktop;
        }

        public BoxCollider Add(Surface surface)
        {
            var go = new GameObject("Surface " + surface.Name);
            go.transform.SetParent(Container.transform, false);
            go.transform.position = surface.Center;
            var box = go.AddComponent<BoxCollider>();
            box.size = surface.Size;
            var material = new PhysicsMaterial("desktop") { dynamicFriction = 0.8f, staticFriction = 0.9f, bounciness = 0f,
                frictionCombine = PhysicsMaterialCombine.Average, bounceCombine = PhysicsMaterialCombine.Minimum };
            box.sharedMaterial = material;
            _surfaces.Add((surface, box));
            return box;
        }

        /// <summary>Removes a surface, e.g. a window closing under her feet.</summary>
        public void Remove(string name)
        {
            for (var i = _surfaces.Count - 1; i >= 0; i--)
            {
                if (_surfaces[i].Surface.Name != name) continue;
                UnityEngine.Object.Destroy(_surfaces[i].Collider.gameObject);
                _surfaces[i].Collider.enabled = false;
                _surfaces.RemoveAt(i);
            }
        }

        /// <summary>
        /// Contact between a collider and the nearest surface: the gap (≥ 0, metres) when apart,
        /// or the penetration depth (≥ 0) when overlapping.
        /// </summary>
        public (float Gap, float Penetration, BoxCollider? Surface) Contact(Collider body)
        {
            var bestGap = float.MaxValue;
            var bestPenetration = 0f;
            BoxCollider? nearest = null;
            foreach (var (_, surface) in _surfaces)
            {
                if (!surface.enabled) continue;
                if (Physics.ComputePenetration(body, body.transform.position, body.transform.rotation,
                        surface, surface.transform.position, surface.transform.rotation, out _, out var depth))
                {
                    if (depth > bestPenetration) { bestPenetration = depth; nearest = surface; }
                    bestGap = 0;
                    continue;
                }
                // Closest points, refined twice (both shapes are convex).
                var p = body.bounds.center;
                var onSurface = surface.ClosestPoint(p);
                var onBody = body.ClosestPoint(onSurface);
                onSurface = surface.ClosestPoint(onBody);
                onBody = body.ClosestPoint(onSurface);
                var gap = Vector3.Distance(onSurface, onBody);
                if (gap < bestGap && bestPenetration == 0) { bestGap = gap; nearest = surface; }
            }
            return (bestPenetration > 0 ? 0 : bestGap, bestPenetration, nearest);
        }

        public void Dispose()
        {
            if (Container != null) UnityEngine.Object.Destroy(Container);
        }
    }
}
