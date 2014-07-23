#pragma once


// =======================================================================================
//                                      ControllerMovementRecorder
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Records the controller angles and transforms for every tick. Provides
///			evaluation functionality of that data as well.
///        
/// # ControllerMovementRecorder
/// 
/// 23-7-2014 Jarl Larsson
///---------------------------------------------------------------------------------------


/*
*
The objective function to be minimized is defined as: fobj(P) = wdfd +wvfv +whfh +wrfr
where fd measures the deviation of the motion from available reference data (m), fv measures
* the average deviation from the desired speed in both sagittal and coronal directions (m/s),
* fh measures head accelerations (m/s2), and fr measures whole body rotations (degrees).
* These terms are weighted usingwd = 100,wv = 5,wh = 0.5,wr = 5. This objective function is used for
* all gaits, although the initial parameter values, initial quadruped state, target velocities,
* gait graph, and reference data will be different for each specific gait.
*
* The fr term measures the
* difference between the desired heading and the actual heading of the character, as measured in
* degrees, by the arc cos(x * ^x), where x and ^x are the actual and desired forward pointing axes of a leg frame.
* This ensures that the quadruped runs forwards rather than sideways. The velocity error term, fv, is
* defined as ||v - vd||, and encompasses both sagittal and coronal directions. v is the mean ve- locity as
* measured over a stride. The desired velocity in the coronal plane is zero. The desired velocity in the
* sagittal plane is an input parameter for the desired gait.

fobj(P) = +wvfv +(whfh) +wrfr
*
* */

class ControllerMovementRecorder
{
public:
	ControllerMovementRecorder()
	{
	}

	~ControllerMovementRecorder()
	{
	}

	double evaluate()
	{
		throw std::exception("The method or operation is not implemented.");
	}

	void fv_calcStrideMeanVelocity(bool p_forceStore = false);

	void fr_calcRotationDeviations();

	void fh_calcHeadAccelerations();

	void fd_calcReferenceMotion();

	void fp_calcMovementDistance();

	double Evaluate();
                      
    // Return standard deviation of fv term
    // as small deviations as possible
	double evaluateFV();

	// mean of FR
	// as small angle difference as possible
	double evaluateFR();

	// mean of FH
	// as small distance as possible
	double evaluateFH();

	double evaluateFD();

	double evaluateFP();


protected:

private:
	/*
	 List<double> m_fvVelocityDeviations = new List<double>(); // (current, mean)-desired
	 List<Vector3> m_fpMovementDist = new List<Vector3>(); // travel distance
	 List<double> m_fhHeadAcceleration = new List<double>();
	 List<double> m_fdBodyHeightSqrDiffs = new List<double>();
	 List<List<float>> m_frBodyRotationDeviations = new List<List<float>>(); //per-leg frame, arcos(current,desired)
	 public float m_fdWeight = 100.0f;
	 public float m_fvWeight = 5.0f;
	 public float m_fhWeight = 0.5f;
	 public float m_frWeight = 5.0f;
	 public float m_fpWeight = 0.5f;
	 float m_origBodyHeight = 0.0f;
	 float m_origHeadHeight = 0.0f;

	 List<Vector3> m_temp_currentStrideVelocities = new List<Vector3>(); // used to calculate mean stride velocity
	 List<Vector3> m_temp_currentStrideDesiredVelocities = new List<Vector3>();
	*/
};


