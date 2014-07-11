#pragma once
#include <glm\gtc\type_ptr.hpp>
// =======================================================================================
//                                      IK2Handler
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Handler and solver for 2-link IK setup
///        
/// # IK2Handler
/// 
/// 11-7-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class IK2Handler
{
public:
	IK2Handler();
	~IK2Handler();


	void solve(const glm::vec3& p_footPosL, const glm::vec3& p_upperLegJointPosL, float p_upperLegLen, float p_lowerLegLen);

protected:
private:


	public LegFrame.LEG m_legType;
	public Transform m_upperLeg;
	public Transform m_lowerLeg;
	public Transform m_foot;
	public Transform m_dbgMesh;
	public LegFrame m_legFrame;

	public float m_hipAngle;
	public float m_kneeAngle;
	public PIDn m_testPIDUpper;
	public PIDn m_testPIDLower;
	public Vector3 m_hipPos;
	public Vector3 m_kneePos;
	public Vector3 m_endPos;
	public Vector3 m_kneePosW;

	private Vector3 m_startPos;
};