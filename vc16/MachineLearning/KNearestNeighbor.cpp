#include "stdafx.h"
#include "KNearestNeighbor.h"

namespace mll
{
	measure::measure() : type(_type), norm(_norm), gain(_gain), offset(_offset), p(_p)
	{
		// Set an object
		setObject();
	}

	measure::~measure()
	{
		// Clear the object
		clearObject();
	}

	void measure::setObject()
	{
		// Set the parameters
		setType(*this);
		_type = KNN_MEASURE_UNKNOWN;
		_norm = false;
		_gain = 1.0;
		_offset = 0.0;
		_p = 1.0;
	}

	void measure::copyObject(const object& obj)
	{
		// Do down casting
		const measure* _obj = static_cast<const measure*>(&obj);

		// Copy the parameters
		_type = _obj->_type;
		_norm = _obj->_norm;
		_gain = _obj->_gain;
		_offset = _obj->_offset;
		_p = _obj->_p;
	}

	void measure::clearObject()
	{

	}

	euclidean::euclidean()
	{
		// Set an object
		setObject();
	}

	euclidean::euclidean(const bool norm, const double gain, const double offset)
	{
		// Set an object
		setObject();

		// Set the parameters
		set(norm, gain, offset);
	}

	euclidean::euclidean(const euclidean& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	euclidean::~euclidean()
	{
		// Clear the object
		clearObject();
	}

	euclidean& euclidean::operator=(const euclidean& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	void euclidean::setObject()
	{
		// Set the parameters
		setType(*this);
		_type = KNN_MEASURE_EUCLIDEAN;
		_norm = false;
		_gain = 1.0;
		_offset = 0.0;
		_p = 2.0;
	}

	void euclidean::set(const bool norm, const double gain, const double offset)
	{
		// Set the parameters
		_norm = norm;
		_gain = gain;
		_offset = offset;
	}

	void euclidean::set(const bool norm, const double gain, const double offset, const double p)
	{
		// Set the parameters
		set(norm, gain, offset);
	}

	const double euclidean::calculate(const numat& xi, const numat& xj) const
	{
		// Calculate a distance between the input dataset
		return pow(numat::sum(numat::pow(numat::abs(xi - xj), p)), 1.0 / p);
	}

	manhattan::manhattan()
	{
		// Set an object
		setObject();
	}

	manhattan::manhattan(const bool norm, const double gain, const double offset)
	{
		// Set an object
		setObject();

		// Set the parameters
		set(norm, gain, offset);
	}

	manhattan::manhattan(const manhattan& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	manhattan::~manhattan()
	{
		// Clear the object
		clearObject();
	}

	manhattan& manhattan::operator=(const manhattan& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	void manhattan::setObject()
	{
		// Set the parameters
		setType(*this);
		_type = KNN_MEASURE_MANHATTAN;
		_norm = false;
		_gain = 1.0;
		_offset = 0.0;
		_p = 1.0;
	}

	minkowski::minkowski()
	{
		// Set an object
		setObject();
	}

	minkowski::minkowski(const bool norm, const double gain, const double offset, const double p)
	{
		// Set an object
		setObject();

		// Set the parameters
		set(norm, gain, offset, p);
	}

	minkowski::minkowski(const minkowski& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	minkowski::~minkowski()
	{
		// Clear the object
		clearObject();
	}

	minkowski& minkowski::operator=(const minkowski& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	void minkowski::setObject()
	{
		// Set the parameters
		setType(*this);
		_type = KNN_MEASURE_MINKOWSKI;
		_norm = false;
		_gain = 1.0;
		_offset = 0.0;
		_p = 1.0;
	}

	void minkowski::set(const bool norm, const double gain, const double offset, const double p)
	{
		// Set the parameters
		_norm = norm;
		_gain = gain;
		_offset = offset;
		_p = p;
	}

	chebychev::chebychev()
	{
		// Set an object
		setObject();
	}

	chebychev::chebychev(const bool norm, const double gain, const double offset)
	{
		// Set an object
		setObject();

		// Set the parameters
		set(norm, gain, offset);
	}

	chebychev::chebychev(const chebychev& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	chebychev::~chebychev()
	{
		// Clear the object
		clearObject();
	}

	chebychev& chebychev::operator=(const chebychev& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	void chebychev::setObject()
	{
		// Set the parameters
		setType(*this);
		_type = KNN_MEASURE_CHEBYCHEV;
		_norm = false;
		_gain = 1.0;
		_offset = 0.0;
	}

	const double chebychev::calculate(const numat& xi, const numat& xj) const
	{
		// Calculate a distance between the input dataset
		return numat::max(numat::abs(xi - xj));
	}

	cosine::cosine()
	{
		// Set an object
		setObject();
	}

	cosine::cosine(const cosine& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	cosine::~cosine()
	{
		// Clear the object
		clearObject();
	}

	cosine& cosine::operator=(const cosine& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	void cosine::setObject()
	{
		// Set the parameters
		setType(*this);
		_type = KNN_MEASURE_COSINE;
		_norm = false;
		_gain = 1.0;
		_offset = 0.0;
	}

	void cosine::set(const bool norm, const double gain, const double offset, const double p)
	{
		// Do nothing
	}

	const double cosine::calculate(const numat& xi, const numat& xj) const
	{
		// Calculate a distance between the input dataset
		return 1.0 - ((xi * xj.T())(0) / sqrt((xi * xi.T())(0) * (xj * xj.T())(0)));
	}

	correlation::correlation()
	{
		// Set an object
		setObject();
	}

	correlation::correlation(const correlation& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	correlation::~correlation()
	{
		// Clear the object
		clearObject();
	}

	correlation& correlation::operator=(const correlation& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	void correlation::setObject()
	{
		// Set the parameters
		setType(*this);
		_type = KNN_MEASURE_CORRELATION;
		_norm = false;
		_gain = 1.0;
		_offset = 0.0;
	}

	KNN::KNN()
	{
		// Set an object
		setObject();
	}

	KNN::KNN(const int K, const measure& meas)
	{
		// Set an object
		setObject();

		// Set a train condition
		condition(K, meas);
	}

	KNN::KNN(const mlldata& dataset, const int K, const measure& meas)
	{
		// Set an object
		setObject();

		// Train the dataset
		train(dataset, K, meas);
	}

	KNN::KNN(const KNN& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	KNN::~KNN()
	{
		// Clear the object
		clearObject();
	}

	KNN& KNN::operator=(const KNN& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	void KNN::setObject()
	{
		// Set the parameters
		setType(*this);
		K = 1;

		// Set the memories
		meas = nullptr;
		minVec.release();
		maxVec.release();
		M.release();
		X.release();
		T.release();
		C.release();
	}

	void KNN::copyObject(const object& obj)
	{
		// Do down casting
		const KNN* _obj = static_cast<const KNN*>(&obj);

		// Copy the parameters
		K = _obj->K;

		// Copy the memories
		copyMeasure(*_obj->meas);
		minVec = _obj->minVec;
		maxVec = _obj->maxVec;
		M = _obj->M;
		X = _obj->X;
		T = _obj->T;
		C = _obj->C;
	}

	void KNN::clearObject()
	{
		// Clear the memories
		if (meas != nullptr)
		{
			delete meas;
		}
		minVec.release();
		maxVec.release();
		M.release();
		X.release();
		T.release();
		C.release();
	}

	void KNN::condition(const int K, const measure& meas)
	{
		// Set a condition
		this->K = K;
		copyMeasure(meas);
	}

	void KNN::createMeasure(const int type, const bool norm, const double gain, const double offset, const double p)
	{
		// Check the old memory
		if (this->meas != nullptr)
		{
			delete this->meas;
		}

		// Create a measurement
		switch (type)
		{
		case KNN_MEASURE_EUCLIDEAN: this->meas = new euclidean(norm, gain, offset); break;
		case KNN_MEASURE_MANHATTAN: this->meas = new manhattan(norm, gain, offset); break;
		case KNN_MEASURE_CHEBYCHEV: this->meas = new chebychev(norm, gain, offset); break;
		case KNN_MEASURE_MINKOWSKI: this->meas = new minkowski(norm, gain, offset, p); break;
		case KNN_MEASURE_COSINE: this->meas = new cosine(); break;
		case KNN_MEASURE_CORRELATION: this->meas = new correlation(); break;
		default: this->meas = nullptr; break;
		}
	}

	void KNN::copyMeasure(const measure& meas)
	{
		// Create a measurement using the input variables
		createMeasure(meas.type, meas.norm, meas.gain, meas.offset, meas.p);
	}

	void KNN::train(const mlldata& dataset)
	{
		// Backup the dataset
		X = dataset[0];			// Feature vector
		T = dataset[1];			// Target vector
		C = dataset[2];			// Class vector

		// Calculate min, max vectors
		minVec = numat::minc(X);
		maxVec = numat::maxc(X);

		// Check the measurement type
		if (meas->type == KNN_MEASURE_CORRELATION)
		{
			// Calculate a mean vector
			M = numat::meanc(X);
		}
		else
		{
			// Initialize a mean vector
			M = numat::zeros(msize(1, X.cols));
		}

		// Check the normalization flag
		if (meas->norm == true)
		{
			convertScale();
		}
	}

	void KNN::train(const mlldata& dataset, const int K, const measure& meas)
	{
		// Set a train condition
		condition(K, meas);

		// Train the dataset
		train(dataset);
	}

	void KNN::convertScale()
	{
		// Convert ccale on the dataset
		for (int i = 0; i < X.cols; i++)
		{
			for (int j = 0; j < X.rows; j++)
			{
				X(j, i) = (X(j, i) - minVec(i)) / (maxVec(i) - minVec(i));
				X(j, i) = meas->gain * X(j, i) + meas->offset;
			}
		}
	}

	const double KNN::predict(const numat& x)
	{
		// Check the normalization flag
		numat xp = x;
		if (meas->norm == true)
		{
			convertScale(xp);
		}

		// Calculate and sort distances
		vector<double> distance;
		vector<double> label;
		for (int i = 0; i < X.rows; i++)
		{
			// Calculate a distance
			double delta = meas->calculate(X.row(i) - M, xp - M);

			// Sort the distances
			bool maxFlag = true;
			for (int j = 0; j < (int)distance.size(); j++)
			{
				if (distance[j] > delta)
				{
					distance.insert(distance.begin() + j, delta);
					label.insert(label.begin() + j, T(i));
					maxFlag = false;
					break;
				}
			}

			// Check the max flag
			if (maxFlag == true)
			{
				distance.push_back(delta);
				label.push_back(T(i));
			}
		}

		// Vote the class using nearest samples
		numem<int> vote(msize(C.length()), 0);
		for (int i = 0; i < K; i++)
		{
			for (int j = 0; j < C.length(); j++)
			{
				if (label[i] == C(j))
				{
					vote(j)++;
				}
			}
		}

		// Get an argument of the maxima label
		int maxValue = vote(0);
		int maxIndex = 0;
		for (int i = 1; i < C.length(); i++)
		{
			if (maxValue < vote(i))
			{
				maxValue = vote(i);
				maxIndex = i;
			}
		}

		return C(maxIndex);
	}

	void KNN::convertScale(numat& x) const
	{
		// Convert scale on the sample data
		for (int i = 0; i < x.length(); i++)
		{
			x(i) = (x(i) - minVec(i)) / (maxVec(i) - minVec(i));
			x(i) = meas->gain * x(i) + meas->offset;
		}
	}

	const int KNN::open(const string path, const string prefix)
	{
		// Open measurement information
		if (openMeasureInfo(path, prefix) != 0)
		{
			return -1;
		}

		// Open label information
		if (openLabelInfo(path, prefix) != 0)
		{
			return -1;
		}

		// Open sample information
		if (openSampleInfo(path, prefix) != 0)
		{
			return -1;
		}

		return 0;
	}

	const int KNN::openMeasureInfo(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("KNN_MEASURE_INFO", prefix)) == false)
		{
			reader.close();

			return -1;
		}

		// Find key name
		vector<string> splitStrs;
		string lineStr;
		string trimStr;
		bool norm = false;
		double gain = 0.0;
		double offset = 0.0;
		double p = 0.0;
		int type = KNN_MEASURE_UNKNOWN;
		while (!reader.eof())
		{
			// Check the end of the section
			getline(reader, lineStr);
			trimStr = trim(lineStr);
			if (trimStr[0] == '[' && trimStr[trimStr.size() - 1] == ']')
			{
				break;
			}

			// Check the string format
			splitStrs = split(trimStr, "=");
			if (splitStrs.size() == 1)
			{
				continue;
			}

			// Check key name
			if (splitStrs[0] == "Normalization")
			{
				if (atoi(splitStrs[1].c_str()) == 1)
				{
					norm = true;
				}
				else
				{
					norm = false;
				}
				continue;
			}
			if (splitStrs[0] == "Gain")
			{
				gain = atof(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Offset")
			{
				offset = atof(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "p")
			{
				p = atof(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Measure_Type")
			{
				type = atoi(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "K")
			{
				K = atoi(splitStrs[1].c_str());
				continue;
			}
		}
		reader.close();

		// Create a measurement for distance calculation
		createMeasure(type, norm, gain, offset, p);

		return 0;
	}

	const int KNN::openLabelInfo(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("KNN_LABEL_INFO", prefix)) == false)
		{
			reader.close();

			return -1;
		}

		// Find key name
		vector<string> splitStrs;
		string lineStr;
		string trimStr;
		while (!reader.eof())
		{
			// Check the end of the section
			getline(reader, lineStr);
			trimStr = trim(lineStr);
			if (trimStr[0] == '[' && trimStr[trimStr.size() - 1] == ']')
			{
				break;
			}

			// Check the string format
			splitStrs = split(trimStr, "=");
			if (splitStrs.size() == 1)
			{
				continue;
			}

			// Check key name
			if (splitStrs[0] == "Num_C")
			{
				C = numat(msize(1, atoi(splitStrs[1].c_str())), 0.0);
				while (!reader.eof())
				{
					// Check the end of the section
					getline(reader, lineStr);
					trimStr = trim(lineStr);
					if (trimStr[0] == '[' && trimStr[trimStr.size() - 1] == ']')
					{
						break;
					}

					// Check a string format
					splitStrs = split(trimStr, "=");
					if (splitStrs.size() == 1)
					{
						continue;
					}

					// Check the key format
					vector<string> indexStrs = split(splitStrs[0], "_");
					if (indexStrs.size() != 2)
					{
						continue;
					}

					// Set a value
					C(atoi(indexStrs[1].c_str())) = atof(splitStrs[1].c_str());
				}
				break;
			}
		}
		reader.close();

		return 0;
	}

	const int KNN::openSampleInfo(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("KNN_SAMPLE_INFO", prefix)) == false)
		{
			reader.close();

			return -1;
		}

		// Find key name
		vector<string> splitStrs;
		string lineStr;
		string trimStr;
		int rows = 0;
		int cols = 0;
		while (!reader.eof())
		{
			// Check the end of the section
			getline(reader, lineStr);
			trimStr = trim(lineStr);
			if (trimStr[0] == '[' && trimStr[trimStr.size() - 1] == ']')
			{
				break;
			}

			// Check the string format
			splitStrs = split(trimStr, "=");
			if (splitStrs.size() == 1)
			{
				continue;
			}

			// Check key name
			if (splitStrs[0] == "Vector_Rows")
			{
				rows = atoi(splitStrs[1].c_str());
				T = numat(msize(rows), 0.0);
				continue;
			}
			if (splitStrs[0] == "Vector_Cols")
			{
				cols = atoi(splitStrs[1].c_str());
				M = numat(msize(1, cols), 0.0);
				minVec = numat(msize(1, cols), 0.0);
				maxVec = numat(msize(1, cols), 0.0);
				X = numat(msize(rows, cols), 0.0);
				while (!reader.eof())
				{
					// Check the end of the section
					getline(reader, lineStr);
					trimStr = trim(lineStr);
					if (trimStr[0] == '[' && trimStr[trimStr.size() - 1] == ']')
					{
						break;
					}

					// Check the string format
					splitStrs = split(trimStr, "=");
					if (splitStrs.size() == 1)
					{
						continue;
					}

					// Check the key format
					vector<string> indexStrs = split(splitStrs[0], "_");
					if (indexStrs.size() == 3)
					{
						if (indexStrs[0] == "Mean" && indexStrs[1] == "Vec")
						{
							M(atoi(indexStrs[2].c_str())) = atof(splitStrs[1].c_str());
						}
						else if (indexStrs[0] == "Min" && indexStrs[1] == "Vec")
						{
							minVec(atoi(indexStrs[2].c_str())) = atof(splitStrs[1].c_str());
						}
						else if (indexStrs[0] == "Max" && indexStrs[1] == "Vec")
						{
							maxVec(atoi(indexStrs[2].c_str())) = atof(splitStrs[1].c_str());
						}
						else if (indexStrs[0] == "Label" && indexStrs[1] == "Vec")
						{
							T(atoi(indexStrs[2].c_str())) = atof(splitStrs[1].c_str());
						}
					}
					else if (indexStrs.size() == 4)
					{
						if (indexStrs[0] == "Sample" && indexStrs[1] == "Vec")
						{
							X(atoi(indexStrs[2].c_str()), atoi(indexStrs[3].c_str())) = atof(splitStrs[1].c_str());
						}
					}
				}
				break;
			}
		}
		reader.close();

		return 0;
	}

	const int KNN::save(const string path, const string prefix)
	{
		// Create a result writer
		ofstream writer(path, ios::trunc);
		if (writer.is_open() == false)
		{
			return -1;
		}

		// Save measurement information
		writer << getSectionName("KNN_MEASURE_INFO", prefix) << endl;
		if (meas->type != KNN_MEASURE_COSINE && meas->type != KNN_MEASURE_CORRELATION)
		{
			if (meas->norm == true)
			{
				writer << "Normalization=1" << endl;
			}
			else
			{
				writer << "Normalization=0" << endl;
			}
			writer << "Gain=" << meas->gain << endl;
			writer << "Offset=" << meas->offset << endl;
			if (meas->type == KNN_MEASURE_MINKOWSKI)
			{
				writer << "p=" << meas->p << endl;
			}
		}
		writer << "Measure_Type=" << (int)meas->type << endl;
		writer << "K=" << K << endl;
		writer << endl;

		// Save label information
		writer << getSectionName("KNN_LABEL_INFO", prefix) << endl;
		writer << "Num_C=" << C.length() << endl;
		for (int i = 0; i < C.length(); i++)
		{
			writer << "C_" << i << "=" << C(i) << endl;
		}
		writer << endl;

		// Save sample information
		writer << getSectionName("KNN_SAMPLE_INFO", prefix) << endl;
		writer << "Vector_Rows=" << X.rows << endl;
		writer << "Vector_Cols=" << X.cols << endl;
		for (int i = 0; i < X.cols; i++)
		{
			writer << "Mean_Vec_" << i << "=" << M(i) << endl;
		}
		for (int i = 0; i < X.cols; i++)
		{
			writer << "Min_Vec_" << i << "=" << minVec(i) << endl;
		}
		for (int i = 0; i < X.cols; i++)
		{
			writer << "Max_Vec_" << i << "=" << maxVec(i) << endl;
		}
		for (int i = 0; i < X.rows; i++)
		{
			for (int j = 0; j < X.cols; j++)
			{
				writer << "Sample_Vec_" << i << "_" << j << "=" << X(i, j) << endl;
			}
		}
		for (int i = 0; i < X.rows; i++)
		{
			writer << "Label_Vec_" << i << "=" << T(i) << endl;
		}
		writer << endl;
		writer.close();

		return 0;
	}
}