using UnityEngine;
using System.Collections;

[System.Serializable]
public struct Axis
{
    // 00 x
    // 01 y
    // 10 z
    bool a, b;

    Vector3 getAxis()
    {
        if (a == false && b == false) return Vector3.right;
        if (a == false && b == true) return Vector3.up;
        if (a == true && b == false) return Vector3.forward;
        return Vector3.zero;
    }
}