using UnityEngine;
using System.Collections;
using System.Collections.Generic;

/*  ===================================================================
 *                             Leg Frame
 *  ===================================================================
 *   The reference frame for a row of legs.
 *   It contains movement properties for this row, as well as feedback
 *   functionalities for torque handling.
 *   Each joint in the leg has a PD controller providing torque.
 *   The hip-joints do not however.
 *   */

public class LegFrame : MonoBehaviour 
{
    public const int c_legCount = 2; // always two legs per frame
    public enum LEG { LEFT = 0, RIGHT = 1 }
    public enum NEIGHBOUR_JOINTS { HIP_LEFT=0, HIP_RIGHT=1, SPINE=2, COUNT=3 }
    public enum PLANE { CORONAL = 0, SAGGITAL = 1 }
    public enum ORIENTATION { YAW = 0, PITCH = 1, ROLL=2 }

    // The gait for each leg
    public StepCycle[] m_tuneStepCycles = new StepCycle[c_legCount];

    // The id of the leg frame in the global torque list
    public int m_id;

    // The joints which are connected directly to the leg frame
    // Ie. the "hip joints" and closest spine segment. 
    // The hip joints are not driven by PD-controllers
    // and will affect the leg frame torque as well as get corrected
    // torques while in stance.
    public int[] m_neighbourJointIds = new int[(int)NEIGHBOUR_JOINTS.COUNT];

    // NOTE!
    // I've lessened the amount of parameters
    // by letting each leg in a leg frame share
    // per-leg parameters. Effectively mirroring
    // behaviour over the saggital plane.
    // There are still two step cycles though, to
    // allow for phase shifting.

    // PLF, the coronal(x) and saggital(y) step distance
    public Vector2 m_tuneStepLength = new Vector2(3.0f, 5.0f);

    // tsw, step interpolation trajectory (horizontal easing between P1 and P2)
    public PcswiseLinear m_tuneStepHorizontalTraj;

    // hsw, step height trajectory
    public PcswiseLinear m_tuneStepHeightTraj;

    // omegaLF, the desired heading orientation trajectory
    // yaw, pitch, roll
    public PcswiseLinear[] m_tuneOrientationLFTraj = new PcswiseLinear[3];

    // PD-controller and driver for calculating desired torque based
    // on the desired orientation
    public PIDn m_desiredLFTorquePD;

    // Foot controllers
    public Vector3[] m_footPlacement = new Vector3[c_legCount];
    public float m_tuneFootPlacementVelocityScale = 1.0f;


    void Awake()
    {
        for (int i = 0; i < (int)NEIGHBOUR_JOINTS.COUNT; i++ )
        {
            m_neighbourJointIds[i] = -1;
        }
        // The orientation heading trajectory starts out
        // without any compensations (flat).
        foreach (PcswiseLinear traj in m_tuneOrientationLFTraj)
        {
            //traj.m_initAsFunc = PcswiseLinear.INITTYPE.FLAT;
            traj.reset();
        }
        
    }

	// Use this for initialization
	void Start () 
    {
	
	}
	
	// Update is called once per frame
	void Update () 
    {
	
	}

    // Calculate the next position where the foot should be placed for legs in swing
    public void updateFootPosForSwingLegs(float p_phi, Vector3 p_velocity, Vector3 p_desiredVelocity)
    {
        for (int i = 0; i < c_legCount; i++)
        {
            // The position is updated as long as the leg
            // is in stance. This means that the last calculated
            // position when the foot leaves the ground is used.
            if (m_tuneStepCycles[i].isInStance(p_phi))
            {
                float mirror=(float)(i*2-1); // flips the coronal axis for the left leg
                Vector3 regularFootPos = transform.TransformPoint(new Vector3(mirror*m_tuneStepLength.x,0.0f,m_tuneStepLength.y));
                Vector3 finalPos=calculateVelocityScaledFootPos(regularFootPos, p_velocity, p_desiredVelocity);
                m_footPlacement[i] = projectFootPosToGround(finalPos);
            }
        }
    }

    private Vector3 projectFootPosToGround(Vector3 p_footPosLF)
    {
        return new Vector3(p_footPosLF.x,0.0f,p_footPosLF.z); // for now, super simple lock to 0
    }

    private Vector3 calculateVelocityScaledFootPos(Vector3 p_footPosLF,
                                                   Vector3 p_velocity,
                                                   Vector3 p_desiredVelocity)
    {
        return p_footPosLF + (p_velocity - p_desiredVelocity) * m_tuneFootPlacementVelocityScale;
    }

    // Retrieves the current orientation quaternion from the
    // trajectory function at time phi.
    private Quaternion getCurrentDesiredOrientation(float p_phi)
    {
        float yaw = m_tuneOrientationLFTraj[(int)ORIENTATION.YAW].getValAt(p_phi);
        float pitch = m_tuneOrientationLFTraj[(int)ORIENTATION.PITCH].getValAt(p_phi);
        float roll = m_tuneOrientationLFTraj[(int)ORIENTATION.ROLL].getValAt(p_phi);
        return Quaternion.Euler(new Vector3(pitch, yaw, roll));
    }

    // Drives the PD-controller and retrieves the 3-axis torque
    // vector that will be used as the desired torque for which the
    // stance legs tries to accomplish.
    private Vector3 getPDTorque(Quaternion p_desiredOrientation)
    {
        Vector3 torque = m_desiredLFTorquePD.drive(transform.rotation,p_desiredOrientation,Time.deltaTime);
        return torque;
    }

    // Function to get the stance and swing legs 
    // sorted into two separate lists
    private void separateLegsPerPhase(float p_phi, 
                                      ref List<int> p_stanceLegs, 
                                      ref List<int> p_swingLegs)
    {
        for (int i = 0; i < c_legCount; i++)
        {
            // Only need to add the indices
            if (m_tuneStepCycles[i].isInStance(p_phi))
            {
                p_stanceLegs.Add(i);
            }
            else
            {
                p_swingLegs.Add(i);
            }
        }
    }

    // This function applies the current torques to the leg frame
    // and corrects the stance leg torques based on a desirec torque for
    // the leg frame itself.
    public Vector3[] applyNetLegFrameTorque(Vector3[] p_currentTorques, float p_phi)
    {
        // Preparations, get ahold of all legs in stance,
        // all legs in swing. And get ahold of their and the 
        // closest spine's torques.
        List<int> stanceLegs=new List<int>();
        List<int> swingLegs=new List<int>();
        Vector3 tstance=Vector3.zero, tswing=Vector3.zero, tspine=Vector3.zero;
        // Find the swing-, and stance legs
        separateLegsPerPhase(p_phi,ref stanceLegs,ref swingLegs);
        // Sum the torques, go through all ids, retrieve their joint id in
        // the global torque vector, and retrieve the current torque:
        for (int i=0;i<stanceLegs.Count;i++)
            tstance+=p_currentTorques[m_neighbourJointIds[stanceLegs[i]]];
        //
        for (int i=0;i<swingLegs.Count;i++)
            tswing+=p_currentTorques[m_neighbourJointIds[swingLegs[i]]];
        //
        int spineIdx=m_neighbourJointIds[(int)NEIGHBOUR_JOINTS.SPINE];
        if (spineIdx!=-1)
            tspine=p_currentTorques[spineIdx];

        // 1. Calculate current torque for leg frame:
        // tLF = tstance + tswing + tspine.
        // Here the desired torque is feedbacked through the
        // stance legs (see 3) as their current torque
        // is the product of previous desired torque combined
        // with current real-world scenarios.
        Vector3 tLF=tstance+tswing+tspine;
        p_currentTorques[m_id]=tLF;

        // 2. Calculate a desired torque, tdLF, using the previous current
        // torque, tLF, and a PD-controller driving towards the 
        // desired orientation, omegaLF.
        Quaternion omegaLF=getCurrentDesiredOrientation(p_phi);
        Vector3 tdLF = getPDTorque(omegaLF);
        // test code
        //rigidbody.AddTorque(tdLF);

        // 3. Now loop through all legs in stance (N) and
        // modify their torques in the vector according
        // to tstancei = (tdLF −tswing −tspine)/N
        // This is to try to make the stance legs compensate
        // for current errors in order to make the leg frame
        // reach its desired torque.
        int N = stanceLegs.Count;
        for (int i = 0; i < N; i++)
        {
            int idx=m_neighbourJointIds[swingLegs[i]];
            p_currentTorques[idx] = (tdLF - tswing - tspine)/(float)N;
            //if (p_currentTorques[idx].magnitude > 100.0f)
            //{
            //    p_currentTorques[idx].Normalize();
            //    Debug.Log("Normalized!");
            //}
            if (float.IsNaN(p_currentTorques[idx].x))
            {
                Debug.Log("NAN");
                Debug.Log("omegaLF "+omegaLF.ToString());
                Debug.Log("current " + transform.rotation.ToString());
                Quaternion error = omegaLF * Quaternion.Inverse(omegaLF);
                // Separate angle and axis, so we can feed the axis-wise
                // errors to the PIDs.
                float a;
                Vector3 dir;
                error.ToAngleAxis(out a, out dir);
                Debug.Log("omegaLF^-1 " + Quaternion.Inverse(omegaLF).ToString());
                Debug.Log("deltaT " + Time.deltaTime);
                Debug.Log("error " + error.ToString());
                Debug.Log("a " + error.ToString());
                Debug.Log("dir " + dir.ToString());
                Debug.Log("tdLF " + tdLF.ToString());
                Debug.Log("tSwing " + tswing.ToString());
                Debug.Log("tSpine " + tspine.ToString());
                Debug.Log(idx + " N: " + N);
                Debug.Log(p_currentTorques[idx]);
                Time.timeScale = 0.0f;
            }
            
        }

        // Return the vector, now containing the new LF torque
        // as well as any corrected stance-leg torques.
        return p_currentTorques;
    }

    public void OnDrawGizmos()
    {
        for (int i = 0; i < m_footPlacement.Length; i++)
        {
            Gizmos.color = Color.green;
            Gizmos.DrawSphere(m_footPlacement[i],0.5f);
        }
    }


}
