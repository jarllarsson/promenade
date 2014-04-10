using UnityEngine;
using System.Collections;
using System.Collections.Generic;


public interface IOptimizable
{
    List<float> GetParams();
    void ConsumeParams(List<float> p_params);
    //double EvaluateFitness();
}
