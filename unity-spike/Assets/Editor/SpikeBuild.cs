using System;
using System.IO;
using System.Linq;
using MotionSpike;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Builds the spike player with SORA's own player settings (SORA/clients/unity/Assets/Editor/
/// SoraBuild.cs): Mono, .NET Standard, Direct3D 11, gamma colour, a Development build, and the
/// runtime VRM shaders always included. So its per-tick numbers transfer to SORA's player.
///   Unity -batchmode -projectPath unity-spike -executeMethod SpikeBuild.Build -quit
/// </summary>
public static class SpikeBuild
{
    private const string ScenePath = "Assets/Scenes/Spike.unity";
    private static readonly string[] Names =
        { "step_128x1", "step_256x1", "step_512x1", "step_128x2", "step_256x2", "step_512x2" };

    public static void Build()
    {
        PlayerSettings.companyName = "motion-lab";
        PlayerSettings.productName = "Motion spike";
        PlayerSettings.runInBackground = true;
        PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
        PlayerSettings.defaultScreenWidth = 540;
        PlayerSettings.defaultScreenHeight = 720;
        PlayerSettings.resizableWindow = true;
        PlayerSettings.usePlayerLog = true;
        PlayerSettings.colorSpace = ColorSpace.Gamma;
        PlayerSettings.SetScriptingBackend(UnityEditor.Build.NamedBuildTarget.Standalone, ScriptingImplementation.Mono2x);
        PlayerSettings.SetApiCompatibilityLevel(UnityEditor.Build.NamedBuildTarget.Standalone, ApiCompatibilityLevel.NET_Standard);
        PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.StandaloneWindows64, false);
        PlayerSettings.SetGraphicsAPIs(BuildTarget.StandaloneWindows64, new[] { GraphicsDeviceType.Direct3D11 });
        var graphics = new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/GraphicsSettings.asset")[0]);
        var shaders = graphics.FindProperty("m_AlwaysIncludedShaders");
        foreach (var name in new[] { "VRM10/MToon10", "UniGLTF/UniUnlit", "Standard" })
        {
            var shader = Shader.Find(name) ?? throw new InvalidOperationException("Shader missing: " + name);
            var present = false;
            for (var i = 0; i < shaders.arraySize; i++)
                if (shaders.GetArrayElementAtIndex(i).objectReferenceValue == shader) present = true;
            if (present) continue;
            shaders.InsertArrayElementAtIndex(shaders.arraySize);
            shaders.GetArrayElementAtIndex(shaders.arraySize - 1).objectReferenceValue = shader;
        }
        graphics.ApplyModifiedPropertiesWithoutUndo();

        Directory.CreateDirectory("Assets/Scenes");
        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
        var player = new GameObject("Spike").AddComponent<SpikePlayer>();
        player.modelNames = Names;
        player.models = Names.Select(n => AssetDatabase.LoadAssetAtPath<ModelAsset>($"Assets/Models/{n}.onnx")
            ?? throw new FileNotFoundException("ModelAsset not imported: " + n)).ToArray();
        EditorSceneManager.SaveScene(scene, ScenePath);
        EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
        AssetDatabase.SaveAssets();

        var result = BuildPipeline.BuildPlayer(new BuildPlayerOptions
        {
            scenes = new[] { ScenePath }, locationPathName = "Builds/Spike/Spike.exe",
            target = BuildTarget.StandaloneWindows64, options = BuildOptions.Development,
        });
        if (result.summary.result != BuildResult.Succeeded)
            throw new InvalidOperationException("Spike build failed: " + result.summary.result);
    }
}
