#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <MathHelp.h>
//#include "CMatrixTest.h"
#include "RandomTest.h"

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

TEST_CASE("Lerp test", "[lerp]") {
	REQUIRE(MathHelp::flerp(0.0f, 1.0f, 0.5f) == Approx(0.5f));
	REQUIRE(MathHelp::flerp(1.0f, 0.0f, 0.5f) == Approx(0.5f));
	REQUIRE(MathHelp::flerp(0.0f, 1.0f, 0.2f) == Approx(0.2f));
	REQUIRE(MathHelp::flerp(0.0f, 1.0f, 0.8f) == Approx(0.8f));
	REQUIRE(MathHelp::flerp(0.0f, 1.0f, 1.0f) == Approx(1.0f));
	REQUIRE(MathHelp::flerp(-10.0f, 10.0f, 0.5f) == Approx(0.0f));
	REQUIRE(MathHelp::flerp(0.0f, 1000.0f, 0.2f) == Approx(200.0f));
	REQUIRE(MathHelp::dlerp(0.0, 1.0, 0.5) == Approx(0.5));
	REQUIRE(MathHelp::dlerp(1.0, 0.0, 0.5) == Approx(0.5));
	REQUIRE(MathHelp::dlerp(0.0, 1.0, 0.2) == Approx(0.2));
	REQUIRE(MathHelp::dlerp(0.0, 1.0, 0.8) == Approx(0.8));
	REQUIRE(MathHelp::dlerp(0.0, 1.0, 1.0) == Approx(1.0));
	REQUIRE(MathHelp::dlerp(-10.0, 10.0, 0.5) == Approx(0.0));
	REQUIRE(MathHelp::dlerp(0.0, 1000.0, 0.2) == Approx(200.0));
}