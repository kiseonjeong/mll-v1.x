#ifndef MLL_UTILITY_H
#define MLL_UTILITY_H

namespace mll
{
	// The Label Position Type
	typedef enum _labelpos
	{
		LABEL_REAR = 0,
		LABEL_FRONT,
		LABEL_EMPTY,
	} labelpos;

	// The Utility for Machine Learning
	class mllutil : public nml::object
	{
		// Variables
	protected:


		// Functions
	protected:
		// Trim a string from the left side
		std::string ltrim(std::string str) const;
		// Trim a string from the right side
		std::string rtrim(std::string str) const;
		// Trim a string from the both sides
		std::string trim(const std::string str) const;
		// Split a string by the separator
		std::vector<std::string> split(const std::string str, const std::string separator) const;

	};
}

#endif