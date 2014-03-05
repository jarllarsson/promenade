using UnityEngine;
using System.Collections;

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
    public enum LEG { LEFT = 0, RIGHT = 1 }
    public enum PLANE { CORONAL = 0, SAGGITAL = 1 }

    // The gait for each leg
    public StepCycle[] m_tuneStepCycles = new StepCycle[2];

    // The joints which are connected directly to the leg frame
    // Ie. the "hip joints". They're not driven by PD-controllers
    // and will affect the leg frame torque as well as get corrected
    // torques while in stance.
    public int[] m_hipJointIds = new int[2];

    // NOTE!
    // I've lessened the amount of parameters
    // by letting each leg in a leg frame share
    // per-leg parameters. Effectively mirroring
    // behaviour over the saggital plane.
    // There are still two step cycles though, to
    // allow for phase shifting.

    // PLF, the coronal(x) and saggital(y) step distance
    public Vector2 m_tuneStepLength = new Vector2(0.0f, 1.0f);

    // tsw, step interpolation trajectory (horizontal easing between P1 and P2)
    public PcswiseLinear m_tuneHorizontalTraj;

    // hsw, step height trajectory
    public PcswiseLinear m_tuneHeightTraj;

    // omegaLF, the desired heading orientation trajectory
    // yaw, pitch, roll
    public PcswiseLinear[] m_orientationTraj = new PcswiseLinear[3];

    // PD-controller and driver for calculating desired torque based
    // on the desired orientation
    public PIDdriverTorque3 m_desiredTorquePD;

    void Awake()
    {
        // The orientation heading trajectory starts out
        // without any compensations (flat).
        foreach (PcswiseLinear traj in m_orientationTraj)
        {
            traj.m_initAsFunc = PcswiseLinear.INITTYPE.FLAT;
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

    private Quaternion getCurrentDesiredOrientation(float p_phi)
    {
        float yaw=m_orientationTraj[0].getValAt(p_phi);
        float pitch=m_orientationTraj[1].getValAt(p_phi);
        float roll=m_orientationTraj[2].getValAt(p_phi);
        return Quaternion.Euler(new Vector3(pitch, yaw, roll));
    }

    private Vector3 getPDTorque(Quaternion p_desiredOrientation)
    {
        Vector3 torque = m_desiredTorquePD.drive(transform.rotation,p_desiredOrientation);
        return torque;
    }

    // This function applies the current torques to the leg frame
    // and corrects the stance leg torques based on a desirec torque for
    // the leg frame itself.
    public Vector3[] applyNetLegFrameTorque(Vector3[] p_currentTorques, float p_phi)
    {
        // 1. Calculate current torque for leg frame:
        // tLF = tstance + tswing + tspine.

        // 2. Calculate a desired torque, tdLF, using the previous current
        // torque, tLF, and a PD-controller driving towards the 
        // desired orientation, omegaLF.
        Quaternion omegaLF=getCurrentDesiredOrientation(p_phi);
        Vector3 tdLF = getPDTorque(omegaLF);
        // test code
        rigidbody.AddTorque(tdLF);

        // 3. Now loop through all legs in stance (N) and
        // modify their torques in the vector according
        // to tstancei = (tdLF −tswing −tspine)/N

        // Return the modified vector
        return p_currentTorques;
    }




}
