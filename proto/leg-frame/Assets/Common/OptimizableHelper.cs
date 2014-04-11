using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public static class OptimizableHelper
{
    public static List<float> ExtractParamsListFrom(Vector2 p_vec2)
    {
        List<float> vals = new List<float>();
        vals.Add(p_vec2.x);
        vals.Add(p_vec2.y);
        return vals;
    }

    public static List<float> ExtractParamsListFrom(Vector3 p_vec3)
    {
        List<float> vals = new List<float>();
        vals.Add(p_vec3.x);
        vals.Add(p_vec3.y);
        vals.Add(p_vec3.z);
        return vals;
    }

    public static List<float> ExtractParamsListFrom(Quaternion p_quat)
    {
        List<float> vals = new List<float>();
        vals.Add(p_quat.x);
        vals.Add(p_quat.y);
        vals.Add(p_quat.z);
        vals.Add(p_quat.w);
        return vals;
    }

    public static void ConsumeParamsTo(List<float> p_params, ref float p_inoutFloat)
    {
        p_inoutFloat = p_params[0];
        p_params.RemoveAt(0);
    }

    public static void ConsumeParamsTo(List<float> p_params, ref Vector2 p_inoutVec2)
    {
        for (int i = 0; i < 2; i++)
        {
            p_inoutVec2[i] = p_params[0];
            p_params.RemoveAt(0);
        }
    }

    public static void ConsumeParamsTo(List<float> p_params, ref Vector3 p_inoutVec3)
    {
        for (int i = 0; i < 3; i++)
        {
            p_inoutVec3[i] = p_params[0];
            p_params.RemoveAt(0);
        }
    }

    public static void ConsumeParamsTo(List<float> p_params, ref Quaternion p_inoutQuat)
    {
        for (int i = 0; i < 4; i++)
        {
            p_inoutQuat[i] = p_params[0];
            p_params.RemoveAt(0);
        }
    }
}
