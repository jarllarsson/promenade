using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class VFChainHandler : MonoBehaviour 
{
    public List<Joint> m_chain=new List<Joint>();
    public List<GameObject> m_chainObjs = new List<GameObject>();
    public List<Vector3> m_torques = new List<Vector3>();
    public Vector3 m_virtualForce;
    public Transform m_target;
    public List<Vector3> m_dofs; // Dofs in link space
    public List<int> m_dofJointId; // Dof's id to joint
	// Use this for initialization
	void Start () 
    {
        for (int i = 0; i < m_chain.Count; i++)
        {
            m_chain[i].m_dofListIdx = m_dofs.Count;
            for (int x = 0; x < m_chain[i].m_dof.Length; x++)
            {
                m_dofs.Add(m_chain[i].m_dof[x]);
                m_dofJointId.Add(i);
            }
        }
	}
	
	// Update is called once per frame
	void Update () 
    {
        //CMatrix J = Jacobian.calculateJacobian(m_chain, m_target.position, Vector3.forward);
        updateChain();
	}

    void FixedUpdate()
    {
        for (int i = 0; i < m_chainObjs.Count; i++)
        {
            m_chainObjs[i].rigidbody.AddRelativeTorque(m_torques[i]);
        }
    }

    void updateChain()
    {
        // Just copy from objects
        Vector3 end = transform.position;
        for (int i = 0; i < m_chain.Count; i++)
        {
            Joint current = m_chain[i];
            GameObject currentObj = m_chainObjs[i];
            current.length = currentObj.transform.localScale.x;
            current.m_position = currentObj.transform.position - currentObj.transform.right * current.length * 0.5f;
            current.m_endPoint = currentObj.transform.position + currentObj.transform.right * current.length * 0.5f;
            end = current.m_endPoint;
        }
        //CMatrix J = Jacobian.calculateJacobian(m_chain, m_chain.Count, end, Vector3.forward);
        CMatrix J = Jacobian.calculateJacobian(m_chain, m_chainObjs, m_dofs, m_dofJointId, end+m_virtualForce);
        CMatrix Jt = CMatrix.Transpose(J);
        CMatrix force = new CMatrix(3, 1);
        force[0, 0] = m_virtualForce.x;
        force[1, 0] = m_virtualForce.y;
        force[2, 0] = m_virtualForce.z;
        //CMatrix torqueSet = Jt*force;
        Debug.Log(Jt.m_rows + "x" + Jt.m_cols);
        //Debug.Log(torqueSet.m_rows+"x"+torqueSet.m_cols);
        //for (int i = 0; i < m_chain.Count; i++)
        //{
        //    // store torque
        //    m_torques[i] = Vector3.forward*Vector3.Dot(new Vector3(Jt[i,0],Jt[i,1],Jt[i,2]),m_virtualForce);
        //    //Debug.Log(m_torques[i].ToString());
        //}

        // This could either be a new kernel, or continuation of the old

        // Here in GPGPU, we could zero the torque position each time...
        for (int i = 0; i < m_chain.Count; i++)
        {
            m_torques[i] = Vector3.zero; // for each dof, for its joint id (as if it had been in the next loop essentially)
        }
        // ... but force sync here, so the following add-op isn't overwritten:
        for (int i = 0; i < m_dofs.Count; i++)
        {
            // store torque
            int x = m_dofJointId[i];
            m_torques[x] += /*m_chainObjs[x].transform.TransformDirection(m_dofs[i])*/m_dofs[i] *Vector3.Dot(new Vector3(Jt[i, 0], Jt[i, 1], Jt[i, 2]), m_virtualForce);
            //Debug.Log(m_torques[i].ToString());
        }
        // Come to think of it, the jacobian and torque could be calculated in the same
        // kernel as it lessens write to global memory and the need to fetch joint matrices several time (transform above)
    }


    void OnDrawGizmos()
    {
        for (int i = 0; i < m_chain.Count; i++)
        {
            Joint joint = m_chain[i];
            Gizmos.color = Color.yellow/**((float)i/(float)m_chain.Count + 0.3f)*/;
            Gizmos.DrawLine(joint.m_position, joint.m_endPoint);
            Gizmos.color = Color.red;
            Gizmos.DrawSphere(joint.m_position, 0.1f);
            Gizmos.DrawLine(joint.m_position, joint.m_position + m_torques[i]);
        }
        Gizmos.color = Color.blue;
        Gizmos.DrawLine(m_chain[m_chain.Count - 1].m_endPoint, m_chain[m_chain.Count - 1].m_endPoint + m_virtualForce);
    }
}
