using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class Jacobian
{
    public static CMatrix calcJ(List<Joint> p_joints, Vector3 p_targetPos)
    {
        int linkCount = p_joints.Count;
        if (linkCount == 0) return null;

        // Construct Jacobian matrix
        CMatrix J = new CMatrix(3, linkCount); // 3 is position in xyz
        for (int i = 0; i < linkCount; i++)
        {
            Vector3 linkPos = p_joints[i].m_position;
            // Currently only solve for z axis(ie. 2d)
            Vector3 rotAxis = new Vector3(0.0f, 0.0f, 1.0f);
            Vector3 dirTarget = Vector3.Cross(rotAxis, p_targetPos - linkPos);
            J[0, i] = dirTarget.x;
            J[1, i] = dirTarget.y;
            J[2, i] = dirTarget.z;
        }

    }
}
