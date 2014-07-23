#include "ControllerMovementRecorder.h"

/*
void fv_calcStrideMeanVelocity(bool p_forceStore=false)
	{
	GaitPlayer player = m_myController.m_player;
	bool restarted = player.checkHasRestartedStride_AndResetFlag();
	if (!restarted && !p_forceStore)
	{
	m_temp_currentStrideVelocities.Add(m_myController.m_currentVelocity);
	m_temp_currentStrideDesiredVelocities.Add(m_myController.m_desiredVelocity);
	}
	else
	{
	Vector3 totalVelocities=Vector3.zero, totalDesiredVelocities=Vector3.zero;
	for (int i = 0; i < m_temp_currentStrideVelocities.Count; i++)
	{
	totalVelocities += m_temp_currentStrideVelocities[i];
	// force straight movement behavior from tests, set desired coronal velocity to constant zero:
	totalDesiredVelocities += new Vector3(0.0f,0.0f,m_temp_currentStrideDesiredVelocities[i].z);
	}
	//Debug.Log("TV: " + totalVelocities + " c " + m_temp_currentStrideVelocities.Count);
	totalVelocities /= (float)m_temp_currentStrideVelocities.Count;
	totalDesiredVelocities /= (float)m_temp_currentStrideDesiredVelocities.Count;
	// add to lists
	//Debug.Log("TV: " + totalVelocities.x + " " + totalVelocities.y + " " + totalVelocities.z);
	m_fvVelocityDeviations.Add((double)Vector3.Magnitude(totalVelocities - totalDesiredVelocities));
	//
	m_temp_currentStrideVelocities.Clear();
	m_temp_currentStrideDesiredVelocities.Clear();
	}
	}

	void fr_calcRotationDeviations()
	{
	int legframes = m_myController.m_legFrames.Length;
	GaitPlayer player = m_myController.m_player;
	for (int i = 0; i < legframes; i++)
	{
	Quaternion currentDesiredOrientation = m_myController.m_legFrames[i].getCurrentDesiredOrientation(player.m_gaitPhase);
	Quaternion currentOrientation = m_myController.m_legFrames[i].transform.rotation;
	Quaternion diff = Quaternion.Inverse(currentOrientation) * currentDesiredOrientation;
	Vector3 axis; float angle;
	diff.ToAngleAxis(out angle,out axis);
	m_frBodyRotationDeviations[i].Add(angle);
	}
	}

	void fh_calcHeadAccelerations()
	{
	m_fhHeadAcceleration.Add((double)m_myController.m_headAcceleration.magnitude);
	}

	void fd_calcReferenceMotion()
	{
	double lenBod = (double)m_myController.transform.position.y - (double)m_origBodyHeight;
	lenBod *= lenBod; // sqr
	double lenHd = (double)m_myController.m_head.transform.position.y - (double)m_origHeadHeight;
	lenHd*= lenHd; // sqr
	double ghostDist = (double)(m_ghostController.position.z - m_ghostStart.z);
	double controllerDist = (double)(m_myController.transform.position.z - m_mycontrollerStart.z);
	double lenDist = ghostDist - controllerDist;
	if (controllerDist < 0.0) lenDist *= 2.0; // penalty for falling or walking backwards
	lenDist *= lenDist; // sqr

	double lenFt = 0.0;
	double lenHips = 0.0;
	double lenKnees = 0.0;
	Vector3 wantedWPos = new Vector3(m_myController.transform.position.x, m_origBodyHeight, m_ghostController.position.z-m_ghostStart.z+m_mycontrollerStart.z);
	wantedWPos = new Vector3(m_myController.transform.position.x, m_origBodyHeight, m_myController.transform.position.z);
	for (int i = 0; i < m_myLegFrame.m_feet.Length; i++)
	{
	Vector3 footRefToFoot   = (m_myLegFrame.m_feet[i].transform.position - wantedWPos) - (m_referenceHandler.m_foot[i].position - m_ghostController.position);
	Vector3 hipRefToHip     = (m_myController.m_joints[(i * 2)+1].transform.position - wantedWPos) - (m_referenceHandler.m_IK[i].m_hipPos - m_ghostController.position);
	Vector3 kneeRefToKnee   = (m_myController.m_joints[(i + 1) * 2].transform.position - wantedWPos) - (m_referenceHandler.m_knee[i].position - m_ghostController.position);
	Debug.DrawLine(m_myLegFrame.m_feet[i].transform.position,
	m_myLegFrame.m_feet[i].transform.position - footRefToFoot, Color.white);
	Debug.DrawLine(m_myController.m_joints[(i * 2) + 1].transform.position,
	m_myController.m_joints[(i * 2) + 1].transform.position - hipRefToHip, Color.yellow*2.0f);
	Debug.DrawLine(m_myController.m_joints[(i + 1) * 2].transform.position,
	m_myController.m_joints[(i + 1) * 2].transform.position - kneeRefToKnee, Color.yellow);

	lenFt += (double)Vector3.SqrMagnitude(footRefToFoot);
	lenHips += (double)Vector3.SqrMagnitude(hipRefToHip);
	lenKnees += (double)Vector3.SqrMagnitude(kneeRefToKnee);
	}

	m_fdBodyHeightSqrDiffs.Add(lenFt * 0.4 + lenKnees + lenHips + lenBod + 2.0f*lenHd + 0.1 * lenDist);
	}

	void fp_calcMovementDistance()
	{
	m_fpMovementDist.Add(m_myController.transform.position - m_myController.m_oldPos);
	}

	public double Evaluate()
	{
	fv_calcStrideMeanVelocity(true);
	double fv = EvaluateFV();
	double fr = EvaluateFR();
	double fh = EvaluateFH();
	double fp = EvaluateFP();
	double fd = EvaluateFD();
	double fobj = (double)m_fdWeight*fd + (double)m_fvWeight*fv + (double)m_frWeight*fr + (double)m_fhWeight*fh - (double)m_fpWeight*fp;
	//Debug.Log(fobj+" = "+(double)m_fvWeight*fv+" + "+(double)m_frWeight*fr+" + "+(double)m_fhWeight*fh+" - "+(double)m_fpWeight*fp);
	return fobj;
	}

	// Return standard deviation of fv term
	// as small deviations as possible
	public double EvaluateFV()
	{
	double total = 0.0f;
	// mean
	foreach (double d in m_fvVelocityDeviations)
	{
	total += d;
	}
	double avg = total /= Math.Max(1.0, ((double)(m_fvVelocityDeviations.Count)));
	double totmeandiffsqr = 0.0f;
	// std
	foreach (double d in m_fvVelocityDeviations)
	{
	double mdiff = d - avg;
	totmeandiffsqr += mdiff * mdiff;
	}
	double sdeviation = Math.Sqrt(totmeandiffsqr / Math.Max(1.0, ((double)m_fvVelocityDeviations.Count)));
	return avg;
	}

	// mean of FR
	// as small angle difference as possible
	public double EvaluateFR()
	{
		double total = 0.0f;
		int sz = 0;
		// mean
		foreach(List<float> fl in m_frBodyRotationDeviations)
			foreach(float f in fl)
		{
			total += (double)f;
			sz++;
		}
		double avg = total /= Math.Max(1, (double)(sz));
		return avg;
	}

	// mean of FH
	// as small distance as possible
	public double EvaluateFH()
	{
		double total = 0.0f;
		int sz = 0;
		// mean
		foreach(double d in m_fhHeadAcceleration)
		{
			total += d;
			sz++;
		}

		double avg = total /= Math.Max(1.0, (double)(sz));
		return avg;
	}

	public double EvaluateFD()
	{
		double total = 0.0f;
		// mean
		foreach(double d in m_fdBodyHeightSqrDiffs)
		{
			total += d;
		}
		double avg = total /= Math.Max(1.0, (double)(m_fdBodyHeightSqrDiffs.Count));
		double totmeandiffsqr = 0.0f;
		// std
		foreach(double d in m_fdBodyHeightSqrDiffs)
		{
			double mdiff = d - avg;
			totmeandiffsqr += mdiff * mdiff;
		}
		double sdeviation = Math.Sqrt(totmeandiffsqr / Math.Max(1.0, (double)m_fdBodyHeightSqrDiffs.Count));
		return avg;
	}

	public double EvaluateFP()
	{
		Vector3 total = Vector3.zero;
		// mean
		foreach(Vector3 dist in m_fpMovementDist)
		{
			total += dist;
		}
		float movementSign = Mathf.Max(0.1f, m_myController.m_goalVelocity.z) / Mathf.Abs(Mathf.Max(0.1f, m_myController.m_goalVelocity.z));
		if (movementSign == 0.0) movementSign = 1.0f;
		double scoreInRightDir = Math.Max(0.0f, total.z * movementSign);
		return scoreInRightDir;
	}
	*/