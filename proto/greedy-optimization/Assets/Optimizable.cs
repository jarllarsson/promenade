using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public interface IOptimizable
{
    List<float> getParams();
    void setParams(List<float> p_params);
}
