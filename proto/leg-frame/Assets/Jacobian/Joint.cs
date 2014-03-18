using UnityEngine;
using System.Collections;

[System.Serializable]
public class Joint
{
    public float length = 1.0f;
    public Vector3 m_angle = Vector3.zero;

    public Vector3 m_position;
    public Vector3 m_endPoint;
    public int m_dofListIdx = 0; // offset in list
    public Vector3[] m_dof; // degrees of freedom, used for init
}
