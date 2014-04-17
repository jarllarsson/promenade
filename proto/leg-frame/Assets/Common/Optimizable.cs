using UnityEngine;
using System.Collections;
using System.Collections.Generic;


public interface IOptimizable
{
    List<float> GetParamsMax();
    List<float> GetParamsMin();
    List<float> GetParams();
    void ConsumeParams(List<float> p_params);
    //double EvaluateFitness();
}
