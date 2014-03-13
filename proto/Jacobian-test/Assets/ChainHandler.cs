using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class ChainHandler : MonoBehaviour 
{
    public List<Joint> m_chain=new List<Joint>();
	// Use this for initialization
	void Start () {
	
	}
	
	// Update is called once per frame
	void Update () 
    {
        updateChain();
	}

    void updateChain()
    {
        Matrix4x4 matParent = Matrix4x4.identity;
        matParent = transform.localToWorldMatrix;
        for (int i = 0; i < m_chain.Count; i++)
        {
            Joint current = m_chain[i];
            Matrix4x4 rotation = Matrix4x4.TRS(Vector3.zero, Quaternion.Euler(current.m_angle), Vector3.one);
            Matrix4x4 len = Matrix4x4.TRS(new Vector3(current.length,0.0f,0.0f), Quaternion.identity, Vector3.one);
            //Debug.Log("L "+len.ToString());
            //Debug.Log("R "+rotation.ToString());
            Matrix4x4 T = matParent * rotation * len;
            //Debug.Log("T " + rotation.ToString());
            current.m_position = new Vector3(matParent[0, 3], matParent[1, 3], matParent[2, 3]);
            current.m_endPoint = new Vector3(T[0, 3], T[1, 3], T[2, 3]);
            matParent = T;
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
    }
}
