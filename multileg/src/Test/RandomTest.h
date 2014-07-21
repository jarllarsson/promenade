#pragma once
#include <Random.h>

TEST_CASE("RandomEngineDeterminism", "[Random]") 
{
	// Test random engine determinism
	Random rnd;
	std::vector<float> list = rnd.getRealUniformList(-0.1f, 1.0f, 10,Random::Generator::DETERMINISTIC);
	REQUIRE(list[0]	== Approx(0.343109250	));
	REQUIRE(list[1]	== Approx(0.647474766	));
	REQUIRE(list[2] == Approx(-0.0657418668	));
	REQUIRE(list[3] == Approx(-0.0495308265	));
	REQUIRE(list[4]	== Approx(0.928523779	));
	REQUIRE(list[5]	== Approx(0.136721581	));
	REQUIRE(list[6]	== Approx(0.609983563	));
	REQUIRE(list[7]	== Approx(0.678050697	));
	REQUIRE(list[8]	== Approx(0.141146272	));
	REQUIRE(list[9]	== Approx(0.826373279	));
}