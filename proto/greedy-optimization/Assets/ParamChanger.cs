using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class ParamChanger
{
    private UniformDistribution m_uniformDistribution;

    public ParamChanger()
    {
        m_uniformDistribution = new UniformDistribution();
        UnityEngine.Random.seed = (int)Time.time;
    }

    public List<float> change(List<float> p_params)
    {
        int size = p_params.Count;
        List<float> result = new List<float>(size);
        List<float> deltaP = getDeltaP(p_params);
        for (int i = 0; i < size; i++)
        {
            result[i] = p_params[i] + deltaP[i];
        }
        return result;
    }

    /// <summary>
    /// Selection vector, determines wether the parameter
    /// at this position in the list should be changed.
    /// 20% probability of change.
    /// </summary>
    /// <param name="p_size"></param>
    /// <returns></returns>
	private List<float> getS(int p_size)
    {
        int changeProbabilityPercent = 20;
        List<float> S = new List<float>(p_size);
        for (int i=0;i<p_size;i++)
        {
            S[i] = UnityEngine.Random.Range(0, 99) < changeProbabilityPercent ? 1.0f : 0.0f;
        }
        return S;
    }

    /// <summary>
    /// Generate DeltaP the change vector to be added to old P.
    /// It contains randomly activated slots. 
    /// Not all parameters will thus be changed by this vector.
    /// </summary>
    /// <param name="p_P"></param>
    /// <returns></returns>
    private List<float> getDeltaP(List<float> p_P)
    {
        int size=p_P.Count;
        // Get R value
        float Pmax = 0.0f, Pmin = 0.0f;
        getMaxMinOfList(p_P, out Pmax, out Pmin);
        float R = Pmax - Pmin;

        // Get S vector
        List<float> S = getS(size);

        // Calculate delta-P
        List<float> deltaP = new List<float>(size);
        for (int i = 0; i < size; i++)
        {
            float P=p_P[i];
            deltaP[i] = S[i] * m_uniformDistribution.U((double)P - 0.1 * (double)R, (double)P + 0.1 * (double)R);
        }
        return deltaP;
    }

    void getMaxMinOfList(List<float> p_list, out float p_min, out float p_max)
    {
        p_min = -999999999.0f;
        p_max = 999999999.0f;
        for (int i=0;i<p_list.Count;i++)
        {
            if (p_list[i] > p_max) p_max = p_list[i];
            if (p_list[i] < p_min) p_min = p_list[i];
        }
    }

   
}
