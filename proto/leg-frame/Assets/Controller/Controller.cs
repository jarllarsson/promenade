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
    public GameObject m_head;
    public ControllerMovementRecorder m_recordedData; // for evaluating controller fitness

    // Desired torques for joints, currently only upper joints(and of course, only during swing for them)
    public PIDn[] m_desiredJointTorquesPD;

    public Vector3 m_oldPos;
    public Vector3 m_currentVelocity;
    public Vector3 m_goalVelocity;
    public Vector3 m_desiredVelocity;
    public float m_debugYOffset = 0.0f;

    private Vector3 m_oldHeadPos;
    private Vector3 m_oldHeadVelocity;
    public Vector3 m_headAcceleration;

    public bool m_usePDTorque = true;
    public bool m_useVFTorque = true;

    public bool m_optimizePDs = false;

    public float m_time = 0.0f;

    void Awake()
    {
        m_oldPos = transform.position;
    }

    // IOptimizable
    public List<float> GetParams()
    {
        List<float> vals = new List<float>();
        for (int i = 0; i < m_legFrames.Length; i++)
            vals.AddRange(m_legFrames[i].GetParams()); // append
        vals.AddRange(m_player.GetParams());

        if (m_optimizePDs)
        {
            for (int i = 1; i < (m_desiredJointTorquesPD.Length + 1) / 2; i++)
            {
                vals.Add(m_desiredJointTorquesPD[i].m_Kp);
                vals.Add(m_desiredJointTorquesPD[i].m_Kd);
            }
        }
        



        return vals;
    }

    public void ConsumeParams(List<float> p_params)
    {
        for (int i = 0; i < m_legFrames.Length; i++)
            m_legFrames[i].ConsumeParams(p_params); // consume
        m_player.ConsumeParams(p_params);

        if (m_optimizePDs)
        {
            for (int i = 1; i < (m_desiredJointTorquesPD.Length + 1) / 2; i++)
            {
                OptimizableHelper.ConsumeParamsTo(p_params, ref m_desiredJointTorquesPD[i].m_Kp);
                m_desiredJointTorquesPD[i + 2].m_Kp = m_desiredJointTorquesPD[i].m_Kp;
                OptimizableHelper.ConsumeParamsTo(p_params, ref m_desiredJointTorquesPD[i].m_Kd);
                m_desiredJointTorquesPD[i + 2].m_Kd = m_desiredJointTorquesPD[i].m_Kd;
            }
        }

    }

    public List<float> GetParamsMax()
    {
        List<float> maxList= new List<float>();
        for (int i = 0; i < m_legFrames.Length; i++)
            maxList.AddRange(m_legFrames[i].GetParamsMax()); // append
        maxList.AddRange(m_player.GetParamsMax());

        if (m_optimizePDs)
        {
            maxList.Add(30);
            maxList.Add(5);

            maxList.Add(30);
            maxList.Add(5);
        }

        return maxList;
    }

    public List<float> GetParamsMin()
    {
        List<float> minList = new List<float>();
        for (int i = 0; i < m_legFrames.Length; i++)
            minList.AddRange(m_legFrames[i].GetParamsMin()); // append
        minList.AddRange(m_player.GetParamsMin());

        if (m_optimizePDs)
        {
            minList.Add(0.01f);
            minList.Add(0.001f);

            minList.Add(0.01f);
            minList.Add(0.001f);
        }

        return minList;
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
            m_joints[i].centerOfMass += -m_joints[i].transform.up * m_joints[i].transform.localScale.y * 0.5f;
            //Debug.Log("COM Joint "+i+" "+m_joints[i].centerOfMass);
            m_chainObjs.Add(m_joints[i].gameObject);
        }

        for (int i = 0; i < m_legFrames.Length; i++)
        {
            m_legFrames[i].OptimizePDs(m_optimizePDs);
        }

        //
        m_oldHeadPos = m_head.transform.position;
        m_oldHeadVelocity = Vector3.zero;
        m_headAcceleration = Vector3.zero;
    }


    // Update is called once per frame
    void Update() 
    {
        m_currentVelocity = transform.position-m_oldPos;
        calcHeadAcceleration();

        m_time += Time.deltaTime;

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
            Vector3 drawTorque = new Vector3(0.0f, 0.0f, -torque.x);
            Debug.DrawLine(m_joints[i].transform.position,m_joints[i].transform.position+drawTorque*0.001f,Color.cyan );
        }
    }

    void calcHeadAcceleration()
    {
        float safeDt = Mathf.Max(0.001f, Time.deltaTime);
        Vector3 headVelocity = (m_head.transform.position - m_oldHeadPos) / safeDt;
        m_headAcceleration = (headVelocity - m_oldHeadVelocity) / safeDt;
        m_oldHeadPos = m_head.transform.position;
        m_oldHeadVelocity = headVelocity;
        Debug.DrawLine(m_head.transform.position, m_head.transform.position + m_headAcceleration*0.001f,Color.blue,0.5f);
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
            m_legFrames[i].updateReferenceFeetPositions(m_player.m_gaitPhase, m_time, m_goalVelocity);
            m_legFrames[i].updateFeet(m_player.m_gaitPhase, m_currentVelocity, m_desiredVelocity);
            m_legFrames[i].tempApplyFootTorque(m_player.m_gaitPhase);
        }
    }

    void updateTorques(float p_dt)
    {
        float phi = m_player.m_gaitPhase;
        // Get the two variants of torque
        Vector3[] tPD = computePDTorques(phi);
        computeCGVFTorques(phi, p_dt);
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
                     if (lf.isInControlledStance(i, m_player.m_gaitPhase))
                     {
                         newTorques[jointID] = Vector3.zero;
                            // m_jointTorques[jointID];
                             //Vector3.zero;
                         //
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
    void computeCGVFTorques(float p_phi, float p_dt)
    {
        if (m_useVFTorque)
        {
            for (int i = 0; i < m_legFrames.Length; i++)
            {
                LegFrame lf = m_legFrames[i];
                // Calculate torques using each leg chain
                for (int n = 1; n < 1+LegFrame.c_legCount*LegFrame.c_legSegments; n++)
                {
                    //  get the joints
                    Rigidbody segment = m_joints[n];
                    lf.calculateFgravcomp(n-1, segment);
                }
            }
        }
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
                    int legFrameRoot = lf.m_id;
                    //legFrameRoot = -1;
                    int legRoot = lf.m_neighbourJointIds[n];
                    int legSegmentCount = LegFrame.c_legSegments; // hardcoded now
                    // Use joint ids to get dof ids
                    // Start in chain
                    int legFrameRootDofId = -1; // if we have separate root as base link
                    if (legFrameRoot!=-1) legFrameRootDofId=m_chain[legFrameRoot].m_dofListIdx;
                    // otherwise, use first in chain as base link
                    int legRootDofId = m_chain[legRoot].m_dofListIdx;
                    // end in chain
                    int lastDofIdx= legRoot + legSegmentCount - 1;
                    int legDofEnd = m_chain[lastDofIdx].m_dofListIdx + m_chain[lastDofIdx].m_dof.Length;
                    //
                    // get force for the leg
                    Vector3 VF = lf.m_netLegBaseVirtualForces[n];
                    // Calculate torques for each joint
                    // Start by updating joint information based on their gameobjects
                    Vector3 end = transform.localPosition;
                    //Debug.Log("legroot "+legRoot+" legseg "+legSegmentCount);
                    int jointstart = legRoot;
                    if (legFrameRoot != -1) jointstart = legFrameRoot;
                    for (int x = jointstart; x < legRoot + legSegmentCount; x++)
                    {
                        if (legFrameRoot != -1 && x<legRoot && x!=legFrameRoot)
                            x = legRoot;
                        Joint current = m_chain[x];
                        GameObject currentObj = m_chainObjs[x];
                        //Debug.Log("joint pos: " + currentObj.transform.localPosition);
                        // Update Joint
                        current.length      = currentObj.transform.localScale.y;
                        current.m_position = currentObj.transform.position /*- (-currentObj.transform.up) * current.length * 0.5f*/;
                        current.m_endPoint = currentObj.transform.position + (-currentObj.transform.up) * current.length/* * 0.5f*/;
                        //m_chain[i] = current;
                        //Debug.DrawLine(current.m_position, current.m_endPoint, Color.red);
                        //Debug.Log(x+" joint pos: " + current.m_position + " = " + m_chain[x].m_position);
                        end = current.m_endPoint;
                    }
                    //foreach(Joint j in m_chain)
                    //    Debug.Log("joint pos CC: " + j.m_position);
                    
                    //CMatrix J = Jacobian.calculateJacobian(m_chain, m_chain.Count, end, Vector3.forward);
                    CMatrix J = Jacobian.calculateJacobian(m_chain,     // Joints (Joint script)
                                                           m_chainObjs, // Gameobjects in chain
                                                           m_dofs,      // Degrees Of Freedom (Per joint)
                                                           m_dofJointId,// Joint id per DOF 
                                                           end + VF,    // Target position
                                                           legRootDofId,// Starting link id in chain (start offset)
                                                           legDofEnd,  // End of chain of link (ie. size)
                                                           legFrameRootDofId); // As we use the leg frame as base, we supply it separately (it will be actual root now)
                    CMatrix Jt = CMatrix.Transpose(J);

                    //Debug.DrawLine(end, end + VF, Color.magenta, 0.3f);
                    int jIdx = 0;
                    int extra = 0;
                    int start = legRootDofId;
                    if (legFrameRootDofId >= 0)
                    {
                        start = legFrameRootDofId;
                        extra = m_chain[legFrameRoot].m_dof.Length;
                    }
                    

                    for (int g = start; g < legDofEnd; g++)
                    {
                        if (extra > 0)
                            extra--;
                        else if (g < legRootDofId)
                            g = legRootDofId;

                        // store torque
                        int x = m_dofJointId[g];
                        Vector3 addT = m_dofs[g] * Vector3.Dot(new Vector3(Jt[jIdx, 0], Jt[jIdx, 1], Jt[jIdx, 2]), VF);
                        newTorques[x] += addT;
                        jIdx++;
                        //Vector3 drawTorque = new Vector3(0.0f, 0.0f, -addT.x);
                        //Debug.DrawLine(m_joints[x].transform.position, m_joints[x].transform.position + drawTorque*0.1f, Color.cyan);

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
                if (lf.isInControlledStance(n, m_player.m_gaitPhase))
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
