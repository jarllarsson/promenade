using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class Jacobian
{
    public static CMatrix calculateJacobian(List<Joint> p_joints, Vector3 p_targetPos, Vector3 p_axis)
    {
        int linkCount = p_joints.Count;
        if (linkCount == 0) return null;

        // Construct Jacobian matrix
        CMatrix J = new CMatrix(3, linkCount); // 3 is position in xyz
        for (int i = 0; i < linkCount; i++)
        {
            Vector3 linkPos = p_joints[i].m_position;
            // Currently only solve for z axis(ie. 2d)
            Vector3 rotAxis = -p_axis;
            Vector3 dirTarget = Vector3.Cross(rotAxis, p_targetPos - linkPos);
            J[0, i] = dirTarget.x;
            J[1, i] = dirTarget.y;
            J[2, i] = dirTarget.z;
        }
        return J;
    }

    public static void updateJacobianTranspose(List<Joint> p_joints, Vector3 p_targetPos, Vector3 p_axis)
    {
        int linkCount = p_joints.Count;
        if (linkCount == 0) return;

        // Calculate Jacobian matrix
        CMatrix J = calculateJacobian(p_joints, p_targetPos, p_axis);

        // Calculate Jacobian transpose
        CMatrix Jt = CMatrix.Transpose(J);

        // Calculate error matrix
        CMatrix e = new CMatrix(3, 1);
        e[0, 0] = p_joints[linkCount - 1].m_endPoint.x - p_targetPos.x;
        e[1, 0] = p_joints[linkCount - 1].m_endPoint.y - p_targetPos.y;
        e[2, 0] = p_joints[linkCount - 1].m_endPoint.z - p_targetPos.z;

        float error = CMatrix.Dot(e, e);
        if (error < 0.0001f)
            return;

        // Calculate mu for inverse estimation
        // ie. a small scalar constant used as step size
        float mudiv = CMatrix.Dot(J * Jt * e, J * Jt * e);
        if (mudiv == 0.0f)
            return;

        float mu = CMatrix.Dot(e, J * Jt * e) / mudiv;

        // Step matrix
        CMatrix deltaAngle = Jt * (mu * e);

        // Make sure the matrix is correct
        if (deltaAngle.m_rows != linkCount)
            Debug.Log("Not correct amount of rows! (" + deltaAngle.m_rows + ") correct is " + linkCount);
        if (deltaAngle.m_cols != 1)
            Debug.Log("Not correct amount of cols! (" + deltaAngle.m_cols + ") correct is 1");

        for (int i = 0; i < linkCount; i++)
        {
            p_joints[i].m_angle += p_axis * deltaAngle[i, 0];
        }
    }
}
