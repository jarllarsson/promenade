using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class Jacobian
{
    public static CMatrix calculateJacobian(List<Joint> p_joints, int p_numberOfLinks, Vector3 p_targetPos, Vector3 p_axis)
    {
        int linkCount = p_numberOfLinks;
        if (linkCount == 0) return null;

        // Construct Jacobian matrix
        CMatrix J = new CMatrix(3, linkCount); // 3 is position in xyz
        for (int i = 0; i < linkCount; i++)
        {
            Vector3 linkPos = p_joints[i].m_position;
            // Currently only solve for z axis(ie. 2d)
            Vector3 rotAxis = p_axis;
            Vector3 dirTarget = Vector3.Cross(rotAxis, p_targetPos - linkPos);
            J[0, i] = dirTarget.x;
            J[1, i] = dirTarget.y;
            J[2, i] = dirTarget.z;
        }
        return J;
    }

    public static CMatrix calculateJacobian(List<Joint> p_joints, List<GameObject> p_jointObjs, 
                                            List<Vector3> p_dofs, List<int> p_dofJointIds, 
                                            Vector3 p_targetPos,
                                            int p_dofListOffset=0, int p_dofListEnd=-1, int p_listStep=1)
    {
        // If GPGPU here
        // First, read dofjoint id, then position from joint array to registry
        //    This is now done in the loop below of course
        // Then also read targetpos, dof into memory
        // The J matrix is in global memory and is written in the end (also do transpose, so really Jt)
        if (p_dofs.Count == 0) return null;

        // This means all Jt's are computed in parallel
        // One Jt per dof

        // Construct Jacobian matrix
        if (p_dofListEnd <= 0) p_dofListEnd = p_dofs.Count;
        CMatrix J = new CMatrix(3, p_dofs.Count); // 3 is position in xyz
        for (int i = p_dofListOffset; i < p_dofListEnd; i += p_listStep) // this is then the "thread pool"
        {
            int id=p_dofJointIds[i];
            Joint joint = p_joints[id];
            Vector3 linkPos = joint.m_position;
            
            // Currently only solve for given axis
            Vector3 rotAxis = p_jointObjs[id].transform.TransformDirection(p_dofs[i]);
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
        CMatrix J = calculateJacobian(p_joints, linkCount, p_targetPos, -p_axis);

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
