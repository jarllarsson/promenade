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
        float[] result = new float[size];
        List<double> deltaP = getDeltaP(p_params);
        for (int i = 0; i < size; i++)
        {
            result[i] = (float)((double)p_params[i] + deltaP[i]);
        }
        return new List<float>(result);
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
        float[] S = new float[p_size];
        for (int i=0;i<p_size;i++)
        {
            S[i] = UnityEngine.Random.Range(0, 99) < changeProbabilityPercent ? 1.0f : 0.0f;
        }
        return new List<float>(S);
    }

    /// <summary>
    /// Generate DeltaP the change vector to be added to old P.
    /// It contains randomly activated slots. 
    /// Not all parameters will thus be changed by this vector.
    /// </summary>
    /// <param name="p_P"></param>
    /// <returns></returns>
    private List<double> getDeltaP(List<float> p_P)
    {
        int size=p_P.Count;
        // Get R value
        double Pmax = 0.0f, Pmin = 0.0f;


       //getMaxMinOfList(p_P, out Pmin, out Pmax);
        double r = 0.001f;
        //r = Random.Range(0.0f, 10.0f);
        Pmax = r; Pmin = -r;


        double R = Pmax - Pmin;
        //Debug.Log("R: " + R + " Pmax: " + Pmax + " Pmin: " + Pmin);

        // Get S vector
        List<float> S = getS(size);

        // Calculate delta-P
        double[] deltaP = new double[size];
        for (int i = 0; i < size; i++)
        {
            double P=(double)p_P[i];
            double c=m_uniformDistribution.U(P - 0.1 * R, P + 0.1 * R);
            deltaP[i] = (double)S[i] * c;
        }
        return new List<double>(deltaP);
    }

    void getMaxMinOfList(List<float> p_list, out double p_min, out double p_max)
    {
        p_min = 999999999.0;
        p_max = 0.0;
        for (int i=0;i<p_list.Count;i++)
        {
            if (p_list[i] > p_max) p_max = p_list[i];
            if (p_list[i] < p_min) p_min = p_list[i];
        }
    }

   
}
