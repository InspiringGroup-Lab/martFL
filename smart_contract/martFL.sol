//SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
library Pairing {
    struct G1Point {
        uint X;
        uint Y;
    }
    // Encoding of field elements is: X[0] * z + X[1]
    struct G2Point {
        uint[2] X;
        uint[2] Y;
    }
    // @return the generator of G1
    function P1() pure internal returns (G1Point memory) {
        return G1Point(1, 2);
    }
    // @return the generator of G2
    function P2() pure internal returns (G2Point memory) {
        return G2Point(
            [10857046999023057135944570762232829481370756359578518086990519993285655852781,
             11559732032986387107991004021392285783925812861821192530917403151452391805634],
            [8495653923123431417604973247489272438418190587263600148770280649306958101930,
             4082367875863433681332203403145435568316851327593401208105741076214120093531]
        );
    }
    // @return the negation of p, i.e. p.addition(p.negate()) should be zero.
    function negate(G1Point memory p) pure internal returns (G1Point memory) {
        // The prime q in the base field F_q for G1
        uint q = 21888242871839275222246405745257275088696311157297823662689037894645226208583;
        if (p.X == 0 && p.Y == 0)
            return G1Point(0, 0);
        return G1Point(p.X, q - (p.Y % q));
    }
    // @return r the sum of two points of G1
    function addition(G1Point memory p1, G1Point memory p2) internal view returns (G1Point memory r) {
        uint[4] memory input;
        input[0] = p1.X;
        input[1] = p1.Y;
        input[2] = p2.X;
        input[3] = p2.Y;
        bool success;
        assembly {
            success := staticcall(sub(gas(), 2000), 6, input, 0xc0, r, 0x60)
            // Use "invalid" to make gas estimation work
            switch success case 0 { invalid() }
        }
        require(success);
    }


    // @return r the product of a point on G1 and a scalar, i.e.
    // p == p.scalar_mul(1) and p.addition(p) == p.scalar_mul(2) for all points p.
    function scalar_mul(G1Point memory p, uint s) internal view returns (G1Point memory r) {
        uint[3] memory input;
        input[0] = p.X;
        input[1] = p.Y;
        input[2] = s;
        bool success;
        assembly {
            success := staticcall(sub(gas(), 2000), 7, input, 0x80, r, 0x60)
            // Use "invalid" to make gas estimation work
            switch success case 0 { invalid() }
        }
        require (success);
    }
    // @return the result of computing the pairing check
    // e(p1[0], p2[0]) *  .... * e(p1[n], p2[n]) == 1
    // For example pairing([P1(), P1().negate()], [P2(), P2()]) should
    // return true.
    function pairing(G1Point[] memory p1, G2Point[] memory p2) internal view returns (bool) {
        require(p1.length == p2.length);
        uint elements = p1.length;
        uint inputSize = elements * 6;
        uint[] memory input = new uint[](inputSize);
        for (uint i = 0; i < elements; i++)
        {
            input[i * 6 + 0] = p1[i].X;
            input[i * 6 + 1] = p1[i].Y;
            input[i * 6 + 2] = p2[i].X[1];
            input[i * 6 + 3] = p2[i].X[0];
            input[i * 6 + 4] = p2[i].Y[1];
            input[i * 6 + 5] = p2[i].Y[0];
        }
        uint[1] memory out;
        bool success;
        assembly {
            success := staticcall(sub(gas(), 2000), 8, add(input, 0x20), mul(inputSize, 0x20), out, 0x20)
            // Use "invalid" to make gas estimation work
            switch success case 0 { invalid() }
        }
        require(success);
        return out[0] != 0;
    }
    // Convenience method for a pairing check for two pairs.
    function pairingProd2(G1Point memory a1, G2Point memory a2, G1Point memory b1, G2Point memory b2) internal view returns (bool) {
        G1Point[] memory p1 = new G1Point[](2);
        G2Point[] memory p2 = new G2Point[](2);
        p1[0] = a1;
        p1[1] = b1;
        p2[0] = a2;
        p2[1] = b2;
        return pairing(p1, p2);
    }
    // Convenience method for a pairing check for three pairs.
    function pairingProd3(
            G1Point memory a1, G2Point memory a2,
            G1Point memory b1, G2Point memory b2,
            G1Point memory c1, G2Point memory c2
    ) internal view returns (bool) {
        G1Point[] memory p1 = new G1Point[](3);
        G2Point[] memory p2 = new G2Point[](3);
        p1[0] = a1;
        p1[1] = b1;
        p1[2] = c1;
        p2[0] = a2;
        p2[1] = b2;
        p2[2] = c2;
        return pairing(p1, p2);
    }
    // Convenience method for a pairing check for four pairs.
    function pairingProd4(
            G1Point memory a1, G2Point memory a2,
            G1Point memory b1, G2Point memory b2,
            G1Point memory c1, G2Point memory c2,
            G1Point memory d1, G2Point memory d2
    ) internal view returns (bool) {
        G1Point[] memory p1 = new G1Point[](4);
        G2Point[] memory p2 = new G2Point[](4);
        p1[0] = a1;
        p1[1] = b1;
        p1[2] = c1;
        p1[3] = d1;
        p2[0] = a2;
        p2[1] = b2;
        p2[2] = c2;
        p2[3] = d2;
        return pairing(p1, p2);
    }
}

contract martFL {
    using Pairing for *;
    struct VerifyingKey {
        Pairing.G1Point alpha;
        Pairing.G2Point beta;
        Pairing.G2Point gamma;
        Pairing.G2Point delta;
        Pairing.G1Point[] gamma_abc;
    }
    struct Proof {
        Pairing.G1Point a;
        Pairing.G2Point b;
        Pairing.G1Point c;
    }

    event LogUint(string,uint);
    function verifyingKey(uint256[] memory vk_input) pure internal returns (VerifyingKey memory vk){
        //emit LogUint("vk_input.length",vk_input.length);
        vk.alpha = Pairing.G1Point(vk_input[0],vk_input[1]);
        vk.beta = Pairing.G2Point([vk_input[2],vk_input[3]],[vk_input[4],vk_input[5]]);
        vk.gamma = Pairing.G2Point([vk_input[6],vk_input[7]],[vk_input[8],vk_input[9]]);
        vk.delta = Pairing.G2Point([vk_input[10],vk_input[11]],[vk_input[12],vk_input[13]]);
        uint len = (vk_input.length -14)/2;
        //emit LogUint("len",len);
        uint index = 13;
        vk.gamma_abc = new Pairing.G1Point[](len);
        for (uint i = 0; i < len; i++) {
            uint a = index + 1;
            uint b = index + 2;
            vk.gamma_abc[i] = Pairing.G1Point(vk_input[a],vk_input[b]);
            index += 2;
        }
        return vk;
    }
    function verify(uint256[] memory vk_input, Proof memory proof, uint[] memory input) internal view returns (uint) {
        uint256 snark_scalar_field = 21888242871839275222246405745257275088548364400416034343698204186575808495617;
        VerifyingKey memory vk = verifyingKey(vk_input);
        require(input.length + 1 == vk.gamma_abc.length);
        //emit LogUint("input.length",input.length);
        //emit LogUint("vk.gamma_abc.length",vk.gamma_abc.length);
        // Compute the linear combination vk_x
        Pairing.G1Point memory vk_x = Pairing.G1Point(0, 0);
        for (uint i = 0; i < input.length; i++) {
            require(input[i] < snark_scalar_field);
            vk_x = Pairing.addition(vk_x, Pairing.scalar_mul(vk.gamma_abc[i + 1], input[i]));
        }
        vk_x = Pairing.addition(vk_x, vk.gamma_abc[0]);
        if(!Pairing.pairingProd4(
             proof.a, proof.b,
             Pairing.negate(vk_x), vk.gamma,
             Pairing.negate(proof.c), vk.delta,
             Pairing.negate(vk.alpha), vk.beta)) return 1;
        return 0;
    }

    struct EP{
        uint depositDP;
        uint depositDA;
        uint ts;
        uint delay;
        uint np;
        uint[] samples;
        mapping (uint => uint) amount;
        mapping (uint => string) model;
        bool isRegister;
        bool isPrepared;
        bool isVerified;
        bool isFailed;
        uint256[] vk_input;
        Proof proof;
        uint[] input;
    }
    EP[] training;
    address dataAcquirer;
    uint numEP = 0;
    uint numDataProvider = 0;
    mapping (address => uint) dataProviderID;
    mapping (address => bool) dataProvider;

    constructor(){
        dataAcquirer = payable(msg.sender);
    }

    modifier OnlyDataAcquirer {
        require(msg.sender == dataAcquirer, "Only the contract owner can call this function");
        _;
    }

    function RegisterModel(uint epoch, string memory h_and_s) public returns (uint){
        require(epoch < numEP);
        require(training[epoch].isRegister == false);
        if((block.timestamp-training[epoch].ts)>training[epoch].delay){
            training[epoch].isRegister = true;
        }else{
            if(dataProvider[msg.sender] == false){
                dataProvider[msg.sender] = true;
                dataProviderID[msg.sender] = numDataProvider;
                numDataProvider++;
            }
            uint ID = dataProviderID[msg.sender];
            training[epoch].np ++;
            training[epoch].model[ID] = h_and_s;
        }
        return block.timestamp-training[epoch].ts;
    }

    function EmptyFunction(uint times) public payable returns (uint){
        times += msg.value;
        return times;
    }

    event NewEpochEvent(uint numEP,uint registerTime);
    event EpochPrepareEvent(uint epoch);
    event EpochVerified(uint epoch,bool result);

    function NewEpoch(uint delay) public payable returns (uint){
        if(numEP > 0){
            require(training[numEP-1].isVerified==true);
        }
        EP storage e = training.push();
        e.depositDP = msg.value/2;
        e.depositDA = msg.value/2;
        e.ts = block.timestamp;
        e.delay = delay;
        e.np = 0;
        e.isRegister = false;
        e.isPrepared = false;
        e.isVerified = false;
        e.isFailed = false;
        numEP ++;
        emit NewEpochEvent(numEP-1,e.ts+e.delay);
        return numEP;
    }

    function ReadEpoch(uint epoch)public view returns (uint,uint,uint,uint,uint,bool,bool,bool,bool){
        require(epoch < numEP);
        EP storage e = training[epoch];
        return (e.depositDA,e.depositDP,e.ts,e.delay,block.timestamp,e.isRegister,e.isPrepared,e.isVerified,e.isFailed);
    }

    function summ(uint[] memory amount)private pure returns (uint){
        uint s = 0;
        for(uint i = 0;i<amount.length;i++){
            s += amount[i];
        }
        return s;
    }

    function Prepare(uint epoch, uint[] memory addr, uint[] memory amount) public OnlyDataAcquirer returns (bool){
        require(epoch < numEP);
        EP storage e = training[epoch];
        require(e.isPrepared == false);
        require(addr.length == amount.length);
        require(summ(amount) == e.depositDP);
        if(block.timestamp>(e.delay+e.ts)){
            training[epoch].isRegister = true;
        }
        
        require(addr.length == e.np);
        for(uint i = 0;i<addr.length;i++){
            training[epoch].amount[addr[i]] =  amount[i];
        }
        training[epoch].samples = [1,2,3,4];
        training[epoch].isPrepared = true;
        emit EpochPrepareEvent(epoch);
        return true;
    }

    event EpochDepositEvent(uint epoch,uint amountDP);

    function DepositEpoch(uint epoch) public payable returns (uint){
        require(epoch<numEP);
        require(training[epoch].isRegister == false);
        if((block.timestamp- training[epoch].ts) > training[epoch].delay){
            training[epoch].isRegister = true;
        }else{
            training[epoch].depositDP += msg.value/2;
            training[epoch].depositDA += msg.value/2;
            emit EpochDepositEvent(epoch,training[epoch].depositDP);
        }
        return training[epoch].depositDA*2;
    }

    function commitInput(uint epoch,uint256[] memory vk_input, Proof memory proof, uint[] memory input)public OnlyDataAcquirer{
        require(training[epoch].isVerified == false && training[epoch].isPrepared == true);
        training[epoch].vk_input = vk_input;
        training[epoch].proof = proof;
        training[epoch].input = input;
    }

    function verifyZkp(uint epoch)public OnlyDataAcquirer{
        require(training[epoch].isVerified == false && training[epoch].isPrepared == true);
        bool v = (verify(training[epoch].vk_input, training[epoch].proof,training[epoch].input) == 0);
        training[epoch].isVerified = true;
        if(v == false){
            training[epoch].isFailed = true;
            for(uint i = 0;i<training[epoch].np;i++){
                training[epoch].amount[i] += training[epoch].depositDA/training[epoch].np;
            }
            training[epoch].depositDA = 0;
        }    
        emit EpochVerified(epoch,v);
    }

    function claim(uint epoch)public payable{
        require(epoch < numEP);
        require(training[epoch].isVerified == true);
        require(msg.sender == dataAcquirer || dataProvider[msg.sender] == true);
        address payable addr = payable(msg.sender);
        if(addr == dataAcquirer){
            uint amount = training[epoch].depositDA;
            if(addr.send(amount)){
                training[epoch].depositDA = 0;
            } 
        }else{
            uint ID = dataProviderID[msg.sender];
            uint amount = training[epoch].amount[ID];
            if (addr.send(amount)) {
                training[epoch].amount[ID] = 0;
            }
        }        
    }
    
}