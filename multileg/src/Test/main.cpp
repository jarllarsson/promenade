#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <MathHelp.h>
#include "CMatrixTest.h"

// =======================================================================================
//                                      Unit Tests
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Project for performing unit tests on solution functionality using CATCH
///        
/// # main
/// 
/// 9-6-2014 Jarl Larsson
///---------------------------------------------------------------------------------------
// 


TEST_CASE("Shortening of vec4 to vec3", "[vec3]") {
	REQUIRE(MathHelp::toVec3(glm::vec4(1.4f, 3.4f, 5.5f, 1.0f)) == glm::vec3(1.4f, 3.4f, 5.5f));
	REQUIRE(MathHelp::toVec3(glm::vec4(-4000.0f, 0.1f, 1.0f, 0.0f)) == glm::vec3(-4000.0f, 0.1f, 1.0f));
	REQUIRE(MathHelp::toVec3(glm::vec4(1000.0f, 0.768768f, 200.2f, 1.0f)) == glm::vec3(1000.0f, 0.768768f, 200.2f));
	REQUIRE(MathHelp::toVec3(glm::vec4(0.0f, 3.4f, 5.5f, 0.0f)) == glm::vec3(0.0f, 3.4f, 5.5f));
}

