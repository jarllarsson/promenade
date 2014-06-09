#pragma once
#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include "Other.h"
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

unsigned int Factorial(unsigned int number) {
	return number <= 1 ? number : Factorial(number - 1)*number;
}

TEST_CASE("Factorials are computed", "[factorial]") {
	REQUIRE(Factorial(1) == 1);
	REQUIRE(Factorial(2) == 2);
	REQUIRE(Factorial(3) == 6);
	REQUIRE(Factorial(10) == 3628800);
}

