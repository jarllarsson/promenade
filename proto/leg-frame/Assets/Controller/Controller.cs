using UnityEngine;
using System.Collections;
using System.Collections.Generic;

/*  ===================================================================
 *                             Controller
 *  ===================================================================
 *   The overall locomotion controller.
 *   Contains the top-level controller logic, which sums all
 *   the torques and feeds them to the physics engine.
 *   */

public class Controller : MonoBehaviour, IOptimizable
{
    public LegFrame[] m_legFrames=new LegFrame[1];
    public GaitPlayer m_player;
    private Vector3[] m_jointTorques; // Maybe separate into joints and leg frames
    public Rigidbody[] m_joints;
    public List<Vector3> m_dofs; // Dofs in link space
    public List<int> m_dofJointId; // Dof's id to joint
    public List<Joint> m_chain;
    public List<GameObject> m_chainObjs;
    public ControllerMovementRecorder m_recordedData; // for evaluating controller fitness

    // Desired torques for joints, currently only upper joints(and of course, only during swing for them)
    public PIDn[] m_desiredJointTorquesPD;

    private Vector3 m_oldPos;
    private Vector3 m_currentVelocity;
    public Vector3 m_goalVelocity;
    private Vector3 m_desiredVelocity;
    public float m_debugYOffset = 0.0f;

    public bool m_usePDTorque = true;
    public bool m_useVFTorque = true;

    // IOptimizable
    public List<float> GetParams()
    {
        List<float> vals = new List<float>();
        for (int i = 0; i < m_legFrames.Length; i++)
            vals.AddRange(m_legFrames[i].GetParams()); // append
        return vals;
    }

    public void ConsumeParams(List<float> p_params)
    {
        for (int i = 0; i < m_legFrames.Length; i++)
            m_legFrames[i].ConsumeParams(p_params); // consume
    }



    void Start()
    {
        m_jointTorques = new Vector3[m_joints.Length];
        // hard code for now
        // neighbour joints
        m_legFrames[0].m_neighbourJointIds[(int)LegFrame.LEG.LEFT] = 1;
        m_legFrames[0].m_neighbourJointIds[(int)LegFrame.LEG.RIGHT] = 3;
        m_legFrames[0].m_id = 0;
        // remaining legs
        if (m_legFrames[0].m_legJointIds.Length>0)
        {
            m_legFrames[0].m_legJointIds[0] = 2;
            m_legFrames[0].m_legJointIds[1] = 4;
        }
        // calculate DOF
        for (int i = 0; i < m_chain.Count; i++)
        {
            m_chain[i].m_dofListIdx = m_dofs.Count;
            for (int x = 0; x < m_chain[i].m_dof.Length; x++)
            {
                m_dofs.Add(m_chain[i].m_dof[x]);
                m_dofJointId.Add(i);
            }
        }
        //
        for (int i=0; i<m_joints.Length; i++)
        {
            m_chainObjs.Add(m_joints[i].gameObject);
        }

        //

    }


    // Update is called once per frame
    void Update() 
    {
        m_currentVelocity = transform.position-m_oldPos;

        // Advance the player
        m_player.updatePhase(Time.deltaTime);

        // Update desired velocity
        updateDesiredVelocity(Time.deltaTime);

        // update feet positions
        updateFeet();

        // Recalculate all torques for this frame
        updateTorques(Time.deltaTime);

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
            //Debug.DrawLine(m_joints[i].transform.position,m_joints[i].transform.position+torque,Color.cyan );
        }
    }

    void LateUpdate()
    {
        // testfix for angular limit
        // UCAM-CL-TR-683.pdf
        //Quaternion parent = m_joints[0].transform.rotation;
        //Quaternion localRot = m_joints[3].transform.rotation * Quaternion.Inverse(parent);
        //float angle=0.0f; Vector3 axis;
        //localRot.ToAngleAxis(out angle, out axis);
        //Vector3 prohibited = Vector3.up;
        //float c = Vector3.Dot(prohibited, -axis);
        //
        //angle = Mathf.Clamp(angle, 320.0f, 350.0f);
        //localRot = Quaternion.AngleAxis(angle, axis);
        //localRot *= parent;
        //m_joints[3].transform.rotation = localRot;
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

    void updateTorques(float p_dt)
    {
        float phi = m_player.m_gaitPhase;
        // Get the two variants of torque
        Vector3[] tPD = computePDTorques(phi);
        Vector3[] tVF = computeVFTorques(phi,p_dt);
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
         if (m_usePDTorque)
         {
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
                     else if (m_desiredJointTorquesPD.Length > 0)
                     {
                         newTorques[jointID] = m_desiredJointTorquesPD[jointID].m_vec;
                     }
                 }
                 // All other joints
                 for (int n = 0; n < lf.m_legJointIds.Length; n++)
                 {
                     int jointID = lf.m_legJointIds[n];
                     if (jointID > -1)
                         newTorques[jointID] = m_desiredJointTorquesPD[jointID].m_vec;
                 }
             }
         }
        return newTorques;
    }


    // Compute the torque of all applied virtual forces
    Vector3[] computeVFTorques(float p_phi, float p_dt)
    {
        Vector3[] newTorques = new Vector3[m_jointTorques.Length];
        if (m_useVFTorque)
        {
            for (int i = 0; i < m_legFrames.Length; i++)
            {
                LegFrame lf = m_legFrames[i];
                lf.calculateNetLegVF(p_phi, p_dt, m_currentVelocity, m_desiredVelocity);
                // Calculate torques using each leg chain
                for (int n = 0; n < LegFrame.c_legCount; n++)
                {
                    //  get the joints
                    int legRoot = lf.m_neighbourJointIds[n];
                    int legSegmentCount = 2; // hardcoded now
                    // Use joint ids to get dof ids
                    int legRootDofId = m_chain[legRoot].m_dofListIdx;
                    int legDofEnd = m_chain[legRoot + legSegmentCount - 1].m_dofListIdx + m_chain[legRoot + legSegmentCount - 1].m_dof.Length;
                    // get force
                    Vector3 VF = lf.m_netLegVirtualForces[n];
                    // Calculate torques for each joint
                    // Just copy from objects
                    Vector3 end = transform.position;
                    for (int x = legRoot; x < legRoot + legSegmentCount; x++)
                    {
                        Joint current = m_chain[i];
                        GameObject currentObj = m_chainObjs[x];
                        current.length = currentObj.transform.localScale.y;
                        current.m_position = currentObj.transform.position /*- (-currentObj.transform.up) * current.length * 0.5f*/;
                        current.m_endPoint = currentObj.transform.position + (-currentObj.transform.up) * current.length/* * 0.5f*/;
                        //Debug.DrawLine(current.m_position, current.m_endPoint, Color.red);
                        
                        end = current.m_endPoint;
                    }
                    //CMatrix J = Jacobian.calculateJacobian(m_chain, m_chain.Count, end, Vector3.forward);
                    CMatrix J = Jacobian.calculateJacobian(m_chain, m_chainObjs, m_dofs, m_dofJointId, end + VF,
                                                           legRootDofId, legDofEnd);
                    CMatrix Jt = CMatrix.Transpose(J);

                    //Debug.DrawLine(end, end + VF, Color.magenta, 0.3f);


                    for (int g = legRootDofId; g < legDofEnd; g++)
                    {
                        // store torque
                        int x = m_dofJointId[g];
                        Vector3 addT = m_dofs[g] * Vector3.Dot(new Vector3(Jt[g, 0], Jt[g, 1], Jt[g, 2]), VF);
                        newTorques[x] += addT;
                        //Debug.DrawLine(m_joints[x].transform.position, m_joints[x].transform.position + addT, Color.blue);

                    }
                    // Come to think of it, the jacobian and torque could be calculated in the same
                    // kernel as it lessens write to global memory and the need to fetch joint matrices several time (transform above)
                }
            }
        }
        return newTorques;
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
