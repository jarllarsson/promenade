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
    private Vector3[] m_jointTorques; // Maybe separate into joints and leg frames
    public Rigidbody[] m_joints;
    // Desired torques for joints, currently only upper joints(and of course, only during swing for them)
    public PIDn[] m_desiredJointTorquesPD;

    private Vector3 m_oldPos;
    private Vector3 m_currentVelocity;
    public Vector3 m_goalVelocity;
    private Vector3 m_desiredVelocity;
    public float m_debugYOffset = 0.0f;


    void Start()
    {
        m_jointTorques = new Vector3[m_joints.Length];
        // hard code for now
        // neighbour joints
        m_legFrames[0].m_neighbourJointIds[(int)LegFrame.LEG.LEFT] = 0;
        m_legFrames[0].m_neighbourJointIds[(int)LegFrame.LEG.RIGHT] = 1;
        m_legFrames[0].m_id = 2;
        // remaining legs
        if (m_legFrames[0].m_legJointIds.Length>0)
        {
            m_legFrames[0].m_legJointIds[0] = 3;
            m_legFrames[0].m_legJointIds[1] = 4;
        }

    }

	
	// Update is called once per frame
	void Update () 
    {
        m_currentVelocity = transform.position-m_oldPos;

        // Advance the player
        m_player.updatePhase(Time.deltaTime);

        // Update desired velocity
        updateDesiredVelocity(Time.deltaTime);

        // update feet positions
        updateFeet();

        // Recalculate all torques for this frame
        updateTorques();

        // Debug color of legs when in stance
        debugColorLegs();

        m_oldPos = transform.position;
	}

    void FixedUpdate()
    {
        for (int i = 0; i < m_jointTorques.Length; i++)
        {
            Vector3 torque = m_jointTorques[i];
            m_joints[i].AddTorque(torque);
            Debug.DrawLine(m_joints[i].transform.position,m_joints[i].transform.position+torque,new Color(i%2,(i%3)*0.5f,(i+2)%4/3.0f) );
        }
    }

    void OnGUI()
    {
        // Draw step cycles
        for (int i = 0; i < m_legFrames.Length; i++)
        {
            drawStepCycles(m_player.m_gaitPhase, m_debugYOffset+10.0f+(float)i*10.0f, m_legFrames[i],i);
        }
    }

    void updateFeet()
    {
        for (int i = 0; i < m_legFrames.Length; i++)
        {
            m_legFrames[i].updateFeet(m_player.m_gaitPhase, m_currentVelocity, m_desiredVelocity);
        }
    }

    void updateTorques()
    {
        float phi = m_player.m_gaitPhase;
        // Get the two variants of torque
        Vector3[] tPD = computePDTorques(phi);
        Vector3[] tVF = computeVFTorques(phi);
        // Sum them
        for (int i = 0; i < m_jointTorques.Length; i++)
        {
            m_jointTorques[i] = tPD[i] + tVF[i];
        }

        // Apply them to the leg frames, also
        // feed back corrections for hip joints
        for (int i = 0; i < m_legFrames.Length; i++)
        {
            m_jointTorques = m_legFrames[i].applyNetLegFrameTorque(m_jointTorques, phi);
        }
    }

    // Compute the torque of all PD-controllers in the joints
    Vector3[] computePDTorques(float p_phi)
    {
         // This loop might have to be rewritten into something a little less cumbersome
         Vector3[] newTorques = new Vector3[m_jointTorques.Length];
         for (int i = 0; i < m_legFrames.Length; i++)
         {
             LegFrame lf = m_legFrames[i];
             newTorques[lf.m_id] = m_jointTorques[lf.m_id];
             // All hip joints
             for (int n = 0; n < lf.m_tuneStepCycles.Length; n++)
             {
                 StepCycle cycle = lf.m_tuneStepCycles[n];
                 int jointID = lf.m_neighbourJointIds[n];
                 if (cycle.isInStance(m_player.m_gaitPhase))
                 {
                     newTorques[jointID] = m_jointTorques[jointID];
                 }
                 else if (m_desiredJointTorquesPD.Length>0)
                 {
                     newTorques[jointID] = m_desiredJointTorquesPD[jointID].m_vec;
                 }
             }
             // All other joints
             for (int n = 0; n < lf.m_legJointIds.Length; n++)
             {
                 int jointID = lf.m_legJointIds[n];
                 newTorques[jointID] = m_desiredJointTorquesPD[jointID].m_vec;
             }
         }
        return newTorques;
    }


    // Compute the torque of all applied virtual forces
    Vector3[] computeVFTorques(float p_phi)
    {
        return new Vector3[m_jointTorques.Length];
    }

    // Function for deciding the current desired velocity in order
    // to reach the goal velocity
    void updateDesiredVelocity(float p_dt)
    {
        float goalSqrMag=m_goalVelocity.sqrMagnitude;
        float currentSqrMag=m_goalVelocity.sqrMagnitude;
        float stepSz = 0.5f * p_dt;
        // Note the material doesn't mention taking dt into 
        // account for the step size, they might be running fixed timestep
        //
        // If the goal is faster
        if (goalSqrMag>currentSqrMag)
        {
            // Take steps no bigger than 0.5m/s
            if (goalSqrMag < currentSqrMag + stepSz)
                m_desiredVelocity=m_goalVelocity;
            else
                m_desiredVelocity += m_currentVelocity.normalized * stepSz;
        }
        else // if the goal is slower
        {
            // Take steps no smaller than 0.5
            if (goalSqrMag > currentSqrMag - stepSz)
                m_desiredVelocity=m_goalVelocity;
            else
                m_desiredVelocity -= m_currentVelocity.normalized * stepSz;
        }
    }

    void debugColorLegs()
    {
        for (int i = 0; i < m_legFrames.Length; i++)
        {
            LegFrame lf = m_legFrames[i];
            for (int n = 0; n < lf.m_tuneStepCycles.Length; n++)
            {
                StepCycle cycle = lf.m_tuneStepCycles[n];
                Rigidbody current = m_joints[lf.m_neighbourJointIds[n]];
                if (cycle.isInStance(m_player.m_gaitPhase))
                {
                    current.gameObject.GetComponentInChildren<Renderer>().material.color = Color.yellow;
                }
                else
                {
                    current.gameObject.GetComponentInChildren<Renderer>().material.color = Color.white;
                }
            }
        }
    }

    void drawStepCycles(float p_phi,float p_yOffset,LegFrame p_frame, int legFrameId)
    {
        for (int i = 0; i < LegFrame.c_legCount; i++)
        {
            StepCycle cycle = p_frame.m_tuneStepCycles[i];
            if (cycle!=null)
            {
                // DRAW!
                float timelineLen = 300.0f;
                float xpad = 10.0f;
                float offset = cycle.m_tuneStepTrigger;
                float len = cycle.m_tuneDutyFactor;
                float lineStart = xpad;
                float lineEnd = lineStart + timelineLen;
                float dutyEnd = lineStart + timelineLen * (offset + len);
                float w = 4.0f;                
                float y = p_yOffset+(float)i*w*2.0f;
                bool stance = cycle.isInStance(p_phi);
                // Draw back
                Color ucol = Color.white*0.5f+new Color((float)(legFrameId%2), (float)(i%2), 1-(float)(i%2),1.0f);
                int h = (int)w / 2;
                Drawing.DrawLine(new Vector2(lineStart-1, y-h-1), new Vector2(lineEnd+1, y-h-1), Color.black, 1);
                Drawing.DrawLine(new Vector2(lineStart-1, y+h), new Vector2(lineEnd+1, y+h), Color.black, 1);
                Drawing.DrawLine(new Vector2(lineStart-1, y-h-1), new Vector2(lineStart-1, y+h+1), Color.black, 1);
                Drawing.DrawLine(new Vector2(lineEnd+1, y-h-1), new Vector2(lineEnd+1, y+h), Color.black, 1);
                Drawing.DrawLine(new Vector2(lineStart, y), new Vector2(lineEnd, y), new Color(1.0f,1.0f,1.0f,1.0f), w);
                // Color depending on stance
                Color currentCol = Color.black;
                float phase = cycle.getStancePhase(p_phi);
                if (stance)
                    currentCol = Color.Lerp(ucol, Color.black, phase*phase);

                // draw df
                Drawing.DrawLine(new Vector2(lineStart + timelineLen * offset, y), new Vector2(Mathf.Min(lineEnd, dutyEnd), y), currentCol, w);
                // draw rest if out of bounds
                if (offset + len > 1.0f)
                    Drawing.DrawLine(new Vector2(lineStart, y), new Vector2(lineStart + timelineLen * (offset + len - 1.0f), y), currentCol, w);

                // Draw current time marker
                Drawing.DrawLine(new Vector2(lineStart + timelineLen * p_phi-1, y), new Vector2(lineStart + timelineLen * p_phi + 3, y),
                    Color.red, w);
                Drawing.DrawLine(new Vector2(lineStart + timelineLen * p_phi, y), new Vector2(lineStart + timelineLen * p_phi + 2, y),
                    Color.green*2, w);
            }
        }
    }
}
