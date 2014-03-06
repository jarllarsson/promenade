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
        //updateTorques();

	}

    void OnGUI()
    {
        // Draw step cycles
        for (int i = 0; i < m_legFrames.Length; i++)
        {
            drawStepCycles(m_player.m_gaitPhase, 10.0f+(float)i*10.0f, m_legFrames[i],i);
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
        return new Vector3[m_jointTorques.Length];
    }

    // Compute the torque of all applied virtual forces
    Vector3[] computeVFTorques(float p_phi)
    {
        return new Vector3[m_jointTorques.Length];
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
                Debug.Log(ucol.ToString());
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
