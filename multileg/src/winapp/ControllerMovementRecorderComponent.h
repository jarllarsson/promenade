#pragma once

#include <Artemis.h>
#include <vector>
#include <glm\gtc\type_ptr.hpp>

class ControllerComponent;
class ControllerSystem;


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

class ControllerMovementRecorderComponent : public artemis::Component
{
public:
	ControllerMovementRecorderComponent();

	~ControllerMovementRecorderComponent()
	{
	}

	double evaluate();

	void fv_calcStrideMeanVelocity(ControllerComponent* p_controller, ControllerSystem* p_system,
		bool p_forceStore = false);

	void fr_calcRotationDeviations(ControllerComponent* p_controller, ControllerSystem* p_system);

	void fh_calcHeadAccelerations(ControllerComponent* p_controller);

	void fd_calcReferenceMotion(ControllerComponent* p_controller);

	void fp_calcMovementDistance(ControllerComponent* p_controller, ControllerSystem* p_system);

                      
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
	
	 std::vector<double> m_fvVelocityDeviations; // (current, mean)-desired
	 glm::vec3 m_fvVelocityGoal;
	 std::vector<glm::vec3> m_fpMovementDist; // travel distance
	 std::vector<double> m_fhHeadAcceleration;
	 std::vector<double> m_fdBodyHeightSqrDiffs;
	 std::vector<std::vector<float>> m_frBodyRotationDeviations; //per-leg frame, arcos(current,desired)
	 float m_fdWeight;
	 float m_fvWeight;
	 float m_fhWeight;
	 float m_frWeight;
	 float m_fpWeight;
	 /*float m_origBodyHeight = 0.0f;
	 float m_origHeadHeight = 0.0f;
*/
	 std::vector<glm::vec3> m_temp_currentStrideVelocities; // used to calculate mean stride velocity
	 std::vector<glm::vec3> m_temp_currentStrideDesiredVelocities;
	
};


