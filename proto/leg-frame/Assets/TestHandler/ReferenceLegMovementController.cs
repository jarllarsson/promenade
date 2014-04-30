using UnityEngine;
using System.Collections;

public class ReferenceLegMovementController : MonoBehaviour 
{
    public IKSolverSimple[] m_IK = new IKSolverSimple[LegFrame.c_legCount];
    public Transform[] m_foot = new Transform[LegFrame.c_legCount];
    private Vector3[] m_oldFootPos = new Vector3[LegFrame.c_legCount];
    public StepCycle[] m_stepCycles = new StepCycle[LegFrame.c_legCount];
    public GaitPlayer m_player;

    // PLF, the coronal(x) and saggital(y) step distance
    public Vector2 m_stepLength = new Vector2(3.0f, 5.0f);

    public PcswiseLinear m_stepHeightTraj;
    private Vector3[] m_liftPos = new Vector3[LegFrame.c_legCount];
    //public PcswiseLinear m_tuneFootTransitionEase;
    
    void Awake()
    {
        for (int i = 0; i < m_foot.Length; i++)
        {
            m_IK[i].m_foot = m_foot[i];
            m_oldFootPos[i] = m_foot[i].position;
        }
    }

    // Use this for initialization
	void Start () 
    {
        for (int i = 0; i < m_foot.Length; i++)
        {
            m_liftPos[i] = m_foot[i].position;
        }
	}
	
	// Update is called once per frame
	void Update () 
    {
        // Advance the player
        m_player.updatePhase(Time.deltaTime);

        updateFeetPositions(m_player.m_gaitPhase);
	}

    void updateFeetPositions(float p_phi)
    {
        for (int i = 0; i < m_foot.Length; i++)
        {
            bool inStance = m_stepCycles[i].isInStance(p_phi);
            //
            if (!inStance)
            {
                float swingPhi = m_stepCycles[i].getSwingPhase(p_phi);
                // The height offset, ie. the "lift" that the foot makes between stepping points.
                Vector3 heightOffset = new Vector3(0.0f, m_stepHeightTraj.getValAt(swingPhi), 0.0f);
                float flip = (i * 2.0f) - 1.0f;
                Vector3 wpos = Vector3.Lerp(m_liftPos[i], 
                                            transform.position + new Vector3(flip*m_stepLength.x,0.0f,m_stepLength.y*0.5f), 
                                            swingPhi);
                wpos = new Vector3(wpos.x, 0.0f, wpos.z);
                m_foot[i].position=wpos+heightOffset;

            }
            else
            {
                m_liftPos[i] = m_foot[i].position;
                Debug.DrawLine(m_foot[i].position, m_foot[i].position+Vector3.up, Color.magenta-new Color(0.3f,0.3f,0.3f,0.0f), 1.0f);
            }
            Color debugColor = Color.red;
            if (i == 1) debugColor = Color.green;
            Debug.DrawLine(m_oldFootPos[i], m_foot[i].position, debugColor, 10.0f);
            m_oldFootPos[i] = m_foot[i].position;
        }
    }


}
