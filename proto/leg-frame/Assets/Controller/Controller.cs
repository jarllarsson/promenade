using UnityEngine;
using System.Collections;

/*  ===================================================================
 *                             Controller
 *  ===================================================================
 *   The overall locomotion controller.
 *   Contains the top-level controller logic, which sums all
 *   the torques and feeds them to the physics engine.
 *   */

public class Controller : MonoBehaviour 
{
    public LegFrame[] m_legFrames=new LegFrame[1];
    public GaitPlayer m_player;
    private Vector3[] m_jointTorques;
    public Rigidbody[] m_joints;

    void Awake()
    {
        m_jointTorques = new Vector3[m_joints.Length];
    }

	// Use this for initialization
	void Start () 
    {
	
	}
	
	// Update is called once per frame
	void Update () 
    {
        // Advance the player
        m_player.updatePhase(Time.deltaTime);

        // Recalculate all torques for this frame
        updateTorques();

	}

    void updateTorques()
    {
        Vector3[] tPD = computePDTorques(m_player.m_gaitPhase);
        Vector3[] tVF = computeVFTorques(m_player.m_gaitPhase);

        for (int i = 0; i < m_jointTorques.Length; i++)
            m_jointTorques[i] = tPD[i] + tVF[i];
    }

    // Compute the torque of all PD-controllers in the joints
    Vector3[] computePDTorques(float p_phi)
    {
        return new Vector3[m_jointTorques.Length];
    }

    // Compute the torque of all applied virtual forces
    Vector3[] computeVFTorques(float p_phi)
    {
        return new Vector3[m_jointTorques.Length];
    }
}
