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
	// Use this for initialization
	void Start () 
    {

	}
	
	// Update is called once per frame
	void Update () 
    {
        //CMatrix J = Jacobian.calculateJacobian(m_chain, m_target.position, Vector3.forward);
        updateChain();
	}

    void updateChain()
    {
        // Just copy from objects
        for (int i = 0; i < m_chain.Count; i++)
        {
            Joint current = m_chain[i];
            GameObject currentObj = m_chainObjs[i];
            current.length = currentObj.transform.localScale.x;
            current.m_position = currentObj.transform.position - currentObj.transform.right * current.length * 0.5f;
            current.m_endPoint = currentObj.transform.position + currentObj.transform.right * current.length * 0.5f;
        }
        CMatrix J = Jacobian.calculateJacobian(m_chain, m_chain.Count, m_target.position, Vector3.forward);
        CMatrix Jt = CMatrix.Transpose(J);
        CMatrix force = new CMatrix(1, 3);
        force[0, 0] = m_virtualForce.x;
        force[1, 0] = m_virtualForce.y;
        force[2, 0] = m_virtualForce.z;
        CMatrix torqueSet = Jt * force;
        for (int i = 0; i < m_chain.Count; i++)
        {
            // Apply torque
            m_torques[i] = new Vector3(torqueSet[i,0],torqueSet[i,1],torqueSet[i,2]);
        }
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
        }
        Gizmos.color = Color.blue;
        Gizmos.DrawLine(transform.position, transform.position+m_virtualForce);
    }
}
