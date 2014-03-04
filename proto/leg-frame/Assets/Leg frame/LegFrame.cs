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
    enum LEG{LEFT=0,RIGHT=1}
    enum PLANE{CORONAL = 0,SAGGITAL = 1}

    // The gait for each leg
    StepCycle[] m_tuneStepCycles = new StepCycle[2];

    // NOTE!
    // I've lessened the amount of parameters
    // by letting each leg in a leg frame share
    // per-leg parameters. Effectively mirroring
    // behaviour over the saggital plane.
    // There are still two step cycles though, to
    // allow for phase shifting.

    // PLF, the coronal(x) and saggital(y) step distance
    Vector2 m_tuneStepLength = new Vector2(0.0f,1.0f);

    // tsw, step interpolation trajectory (horizontal easing between P1 and P2)
    PcswiseLinear m_tuneHorizontalTraj;

    // hsw, step height trajectory
    PcswiseLinear m_tuneHeightTraj;

	// Use this for initialization
	void Start () 
    {
	
	}
	
	// Update is called once per frame
	void Update () 
    {
	
	}

    //public 




}
