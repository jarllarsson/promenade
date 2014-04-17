using UnityEngine;
using System.Collections;

public class UniformDistribution
{
    static System.Random p_fixedrand;
    public UniformDistribution()
    {
        if (p_fixedrand==null)
            p_fixedrand = new System.Random(4350809);
    }

    public double U(double p_lower, double p_upper)
    {
        double u = p_fixedrand.NextDouble() * (p_upper - p_lower) + p_lower;
        return u;
    }
}
