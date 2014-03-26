using UnityEngine;
using System.Collections;
using System.Collections.Generic;


public interface IOptimizable
{
    List<float> GetParams();
    void SetParams(List<float> p_params);
    double EvaluateFitness();
}
